import wandb
import os
from typing import List, Dict, Any


def get_wandb_runs(entity: str, project: str, per_page: int = 50, page: int = 1, order: str = "-created_at", filters: Dict[str, Any] = None, search: str = None) -> List[Dict[str, Any]]:
    """
    Fetch all runs from a specific wandb entity and project.
    
    Args:
        entity (str): The wandb entity (username or team name)
        project (str): The project name
        per_page (int): Number of runs to fetch per page (default: 50)
        page (int): Page number to fetch (default: 1)
        order (str): Sort order (default: newest first)
        filters (Dict[str, Any]): Dictionary of filters
        search (str): Search string for name/tags
        
    Returns:
        List[Dict[str, Any]]: List of run dictionaries containing run information
    """
    # Initialize wandb API
    api = wandb.Api()
    
    # Get all runs from the specified entity and project
    runs = api.runs(f"{entity}/{project}", per_page=per_page, page=page)
    
    # Convert runs to a list of dictionaries
    runs_data = []
    for run in runs:
        run_dict = {
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "config": run.config,
            "summary": run.summary,
            "created_at": run.created_at,
            "url": run.url,
        }
        runs_data.append(run_dict)
    
    # Apply filters and sorting
    if filters:
        runs_data = [run for run in runs_data if all(run[key] == value for key, value in filters.items())]
    if search:
        runs_data = [run for run in runs_data if search in run['name'] or search in run['tags']]
    runs_data.sort(key=lambda run: run[order.lstrip('-')], reverse=(order.startswith('-')))
    
    return runs_data

def get_run_training_metrics(entity: str, project: str, run_id: str) -> Dict[str, List[Any]]:
    """
    Fetch training metrics history for a specific run.
    
    Args:
        entity (str): The wandb entity (username or team name)
        project (str): The project name
        run_id (str): The ID of the run to fetch metrics for
        
    Returns:
        Dict[str, List[Any]]: Dictionary mapping metric names to their history
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    
    # Get the history of all metrics
    history = run.history()
    
    # Convert to a more convenient format
    metrics = {}
    for column in history.columns:
        if column not in ['_timestamp', '_runtime', '_step']:
            metrics[column] = history[column].tolist()
    
    return metrics

def get_run_system_metrics(entity: str, project: str, run_id: str) -> Dict[str, List[Any]]:
    """
    Fetch system metrics history for a specific run.
    
    Args:
        entity (str): The wandb entity (username or team name)
        project (str): The project name
        run_id (str): The ID of the run to fetch metrics for
        
    Returns:
        Dict[str, List[Any]]: Dictionary mapping system metric names to their history
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    
    # Get the history of system metrics
    system_metrics = run.history(stream="events")
    
    # Convert to a more convenient format
    metrics = {}
    for column in system_metrics.columns:
        if column not in ['_timestamp', '_runtime', '_step']:
            metrics[column] = system_metrics[column].tolist()
    
    return metrics

def get_run_summary_metrics(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """
    Fetch summary metrics for a specific run.
    
    Args:
        entity (str): The wandb entity (username or team name)
        project (str): The project name
        run_id (str): The ID of the run to fetch metrics for
        
    Returns:
        Dict[str, Any]: Dictionary containing summary metrics
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    return run.summary

def main():
    # You can set these as environment variables or hardcode them
    entity = "l2k2" # os.getenv("WANDB_ENTITY")
    project = "finetune" # os.getenv("WANDB_PROJECT")
    
    if not entity or not project:
        print("Please set WANDB_ENTITY and WANDB_PROJECT environment variables")
        return
    
    try:
        runs = get_wandb_runs(entity, project)
        print(f"Found {len(runs)} runs in {entity}/{project}")
        
        # Print basic information about each run
        for run in runs:
            print(f"\nRun ID: {run['id']}")
            print(f"Name: {run['name']}")
            print(f"State: {run['state']}")
            print(f"Created at: {run['created_at']}")
            print(f"URL: {run['url']}")
            
            # Get and print summary metrics
            print("\nSummary Metrics:")
            summary = get_run_summary_metrics(entity, project, run['id'])
            for key, value in summary.items():
                if key.startswith("system/"):
                    print(f"{key}: {value}")
            
            # Get and print training metrics
            print("\nTraining Metrics:")
            training_metrics = get_run_training_metrics(entity, project, run['id'])
            for metric_name, values in training_metrics.items():
                print(f"{metric_name}: {len(values)} values")
                if values:
                    print(f"  Latest value: {values[-1]}")
            
            # Get and print system metrics
            print("\nSystem Metrics History:")
            system_metrics = get_run_system_metrics(entity, project, run['id'])
            for metric_name, values in system_metrics.items():
                print(f"{metric_name}: {len(values)} values")
                if values:
                    print(f"  Latest value: {values[-1]}")
            
    except Exception as e:
        print(f"Error fetching runs: {str(e)}")

if __name__ == "__main__":
    main() 