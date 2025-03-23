# HIQL Push Environment

Changes are made in `impls` folder

## New Files / Modified Files
- `main.py`
- `evaluation.py`
- `dataset_utils.py`
- `task_utils.py`

## Flow of the Pipeline

1. Create the dataset as per environment
2. Create the tasks
   - Each task consists of:
     - Random initial state
     - Final state up until some trajectory length
   - At evaluation:
     - Get action from actor
     - Take a step in environment
     - Update next observation for actor
     - Loop continues until either:
       - Agent reaches desired state
       - Max steps for agent are reached

3. Start training (no modification)
4. At each evaluation interval:
   - Evaluate each task one by one
