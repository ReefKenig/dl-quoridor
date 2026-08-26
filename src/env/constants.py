"""Constants shared across layers. Kept import-light so the model layer never
needs to import the env module just to know the action-space layout."""

# The 12 pawn-move actions (4 orthogonal + 4 jump-straight + 4 jump-diagonal)
# that precede wall actions in the action space; shared by env and model.
NUM_MOVE_ACTIONS = 12
