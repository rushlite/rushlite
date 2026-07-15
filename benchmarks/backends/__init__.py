"""Backend implementations of the common Backend ABC.

Each module registers itself via register_backend() when imported. run.py imports
these lazily so a rushlite-only run never needs torch installed, and vice versa.
"""
