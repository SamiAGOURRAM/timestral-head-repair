"""Step 2: baseline diagnostic score of the unedited model (4 passes)."""
from harness import Bench, show
b = Bench()
print(show(b.score(b.fresh(), "baseline")))
