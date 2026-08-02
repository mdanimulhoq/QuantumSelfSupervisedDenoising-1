"""
IBMQ hardware interface with rate-limiting, retry, and caching.

Implements TDD §7.3 Phase C.
"""

import os
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime import Session
from qiskit_aer import AerSimulator


class IBMQHardware:
    """
    Wrapper for IBMQ hardware access with:
    - Rate limiting (respect QPU time)
    - Retry logic (handle failed jobs)
    - Caching (avoid re-running identical circuits)
    """
    
    def __init__(
        self,
        backend_name: str = "ibm_nairobi",
        cache_dir: Path = Path("data/hardware_cache"),
        max_retries: int = 3,
        rate_limit_seconds: int = 60,
        use_cache: bool = True,
    ):
        """
        Args:
            backend_name: IBMQ backend name (e.g., 'ibm_nairobi')
            cache_dir: Directory for caching results
            max_retries: Maximum number of retries for failed jobs
            rate_limit_seconds: Minimum seconds between QPU submissions
            use_cache: Whether to use cached results
        """
        self.backend_name = backend_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.rate_limit_seconds = rate_limit_seconds
        self.use_cache = use_cache
        
        self._service = None
        self._backend = None
        self._last_submission_time = 0
        self._cache_hits = 0
        self._cache_misses = 0
    
    @property
    def service(self) -> QiskitRuntimeService:
        """Lazy initialize IBMQ service."""
        if self._service is None:
            try:
                self._service = QiskitRuntimeService()
            except Exception as e:
                print(f"⚠️ IBMQ service not available: {e}")
                print("   Using Aer simulator instead.")
                self._service = None
        return self._service
    
    @property
    def backend(self):
        """Lazy initialize backend."""
        if self._backend is None and self.service is not None:
            try:
                self._backend = self.service.backend(self.backend_name)
            except Exception as e:
                print(f"⚠️ Backend {self.backend_name} not available: {e}")
                print("   Using Aer simulator instead.")
                self._backend = None
        return self._backend
    
    def _get_cache_key(self, circuit: QuantumCircuit, shots: int) -> str:
        """Generate cache key from circuit and shots."""
        # Serialize circuit
        circuit_str = circuit.qasm()
        key_str = f"{circuit_str}_{shots}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.json"
    
    def _save_to_cache(self, cache_key: str, counts: Dict[str, int]) -> None:
        """Save counts to cache."""
        cache_path = self._get_cache_path(cache_key)
        data = {
            'counts': counts,
            'timestamp': datetime.now().isoformat(),
            'backend': self.backend_name,
        }
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, int]]:
        """Load counts from cache."""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                data = json.load(f)
                self._cache_hits += 1
                return data['counts']
        self._cache_misses += 1
        return None
    
    def _wait_for_rate_limit(self) -> None:
        """Wait if needed to respect rate limit."""
        current_time = time.time()
        elapsed = current_time - self._last_submission_time
        if elapsed < self.rate_limit_seconds:
            wait_time = self.rate_limit_seconds - elapsed
            print(f"⏳ Rate limiting: waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        self._last_submission_time = time.time()
    
    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        use_hardware: bool = True,
        retry: bool = True,
    ) -> Dict[str, int]:
        """
        Run a circuit on hardware or simulator.
        
        Args:
            circuit: QuantumCircuit to run
            shots: Number of shots
            use_hardware: Whether to use real hardware (vs simulator)
            retry: Whether to retry on failure
        
        Returns:
            Counts dictionary {bitstring: count}
        """
        # Check cache first
        cache_key = self._get_cache_key(circuit, shots)
        if self.use_cache:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                print(f"   💾 Cache hit: {cache_key[:8]}")
                return cached
        
        # Run on hardware or simulator
        if use_hardware and self.backend is not None:
            try:
                return self._run_hardware(circuit, shots, retry)
            except Exception as e:
                print(f"⚠️ Hardware error: {e}")
                print("   Falling back to simulator...")
                return self._run_simulator(circuit, shots)
        else:
            return self._run_simulator(circuit, shots)
    
    def _run_hardware(
        self,
        circuit: QuantumCircuit,
        shots: int,
        retry: bool,
    ) -> Dict[str, int]:
        """Run on real hardware with retry logic."""
        if self.backend is None:
            raise RuntimeError("No hardware backend available")
        
        # Transpile for hardware
        transpiled = transpile(circuit, self.backend)
        
        # Rate limiting
        self._wait_for_rate_limit()
        
        # Submit job
        print(f"   🔬 Submitting to {self.backend_name}...")
        
        attempts = 0
        while attempts < (self.max_retries if retry else 1):
            try:
                with Session(service=self.service, backend=self.backend) as session:
                    sampler = Sampler(session=session)
                    job = sampler.run([transpiled], shots=shots)
                    result = job.result()
                    counts = result[0].data.meas.get_counts()
                    
                    # Cache results
                    if self.use_cache:
                        self._save_to_cache(self._get_cache_key(circuit, shots), dict(counts))
                    
                    return dict(counts)
            except Exception as e:
                attempts += 1
                if attempts < self.max_retries and retry:
                    wait = 2 ** attempts  # Exponential backoff
                    print(f"   ⚠️ Attempt {attempts} failed, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise e
        
        raise RuntimeError("Failed to run on hardware")
    
    def _run_simulator(self, circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
        """Run on Aer simulator."""
        simulator = AerSimulator(shots=shots)
        result = simulator.run(circuit).result()
        counts = result.get_counts()
        
        # Cache results
        if self.use_cache:
            self._save_to_cache(self._get_cache_key(circuit, shots), dict(counts))
        
        return dict(counts)
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'total': self._cache_hits + self._cache_misses,
            'hit_rate': self._cache_hits / (self._cache_hits + self._cache_misses) 
                        if (self._cache_hits + self._cache_misses) > 0 else 0,
        }
    
    def clear_cache(self) -> None:
        """Clear the cache directory."""
        for file in self.cache_dir.glob("*.json"):
            file.unlink()
        print("   🗑️ Cache cleared")


def create_ibmq_wrapper(
    backend_name: str = "ibm_nairobi",
    cache_dir: Path = Path("data/hardware_cache"),
    max_retries: int = 3,
    rate_limit_seconds: int = 60,
    use_cache: bool = True,
) -> IBMQHardware:
    """Convenience function to create IBMQ wrapper."""
    return IBMQHardware(
        backend_name=backend_name,
        cache_dir=cache_dir,
        max_retries=max_retries,
        rate_limit_seconds=rate_limit_seconds,
        use_cache=use_cache,
    )
