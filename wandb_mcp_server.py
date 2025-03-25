from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any
import wandb

# Create an MCP server
mcp = FastMCP("Wandb MCP Server")

@mcp.tool()
def execute_graphql_query(query: str, variables: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Execute an arbitrary GraphQL query against the wandb API.
    
    Args:
        query (str): The GraphQL query string
        variables (Dict[str, Any], optional): Variables to pass to the query
        
    Returns:
        Dict[str, Any]: The query result
        
    Example:
        query = '''
        query Project($entity: String!, $name: String!) {
            project(entityName: $entity, name: $name) {
                name
                entity
                description
                runs {
                    edges {
                        node {
                            id
                            name
                            state
                        }
                    }
                }
            }
        }
        '''
        variables = {
            "entity": "my-entity",
            "name": "my-project"
        }
        result = execute_graphql_query(query, variables)
    """
    # Initialize wandb API
    api = wandb.Api()
    
    # Execute the query
    result = api.client.execute(query, variables or {})
    
    return result

@mcp.tool()
def get_entity_projects(entity: str) -> List[Dict[str, Any]]:
    """
    Fetch all projects for a specific wandb entity.
    
    Args:
        entity (str): The wandb entity (username or team name)
        
    Returns:
        List[Dict[str, Any]]: List of project dictionaries containing:
            - name: Project name
            - entity: Entity name
            - description: Project description
            - visibility: Project visibility (public/private)
            - created_at: Creation timestamp
            - updated_at: Last update timestamp
            - tags: List of project tags
    """
    # Initialize wandb API
    api = wandb.Api()
    
    # Get all projects for the entity
    projects = api.projects(entity)
    
    # Convert projects to a list of dictionaries
    projects_data = []
    for project in projects:
        project_dict = {
            "name": project.name,
            "entity": project.entity,
            "description": project.description,
            "visibility": project.visibility,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            "tags": project.tags,
        }
        projects_data.append(project_dict)
    
    return projects_data

@mcp.tool()
def get_wandb_runs(
    entity: str,
    project: str,
    per_page: int = 50,
    order: str = "-created_at",
    filters: Dict[str, Any] = None,
    search: str = None
) -> List[Dict[str, Any]]:
    """
    Fetch runs from a specific wandb entity and project with filtering and sorting support.
    
    Args:
        entity (str): The wandb entity (username or team name)
        project (str): The project name
        per_page (int): Number of runs to fetch (default: 50)
        order (str): Sort order (default: "-created_at"). Prefix with "-" for descending order.
                    Examples: "created_at", "-created_at", "name", "-name", "state", "-state"
        filters (Dict[str, Any]): Dictionary of filters to apply. Keys can be:
            - state: "running", "finished", "crashed", "failed", "killed"
            - tags: List of tags to filter by
            - config: Dictionary of config parameters to filter by
            - summary: Dictionary of summary metrics to filter by
        search (str): Search string to filter runs by name or tags
        
    Returns:
        List[Dict[str, Any]]: List of run dictionaries containing run information
    """
    # Initialize wandb API
    api = wandb.Api()
    
    # Build query parameters
    query_params = {
        "per_page": per_page,
        "order": order
    }
    
    # Add filters if provided
    if filters:
        for key, value in filters.items():
            if key in ["state", "tags", "config", "summary"]:
                query_params[key] = value
    
    # Add search if provided
    if search:
        query_params["search"] = search
    
    # Get runs from the specified entity and project with filters
    runs = api.runs(
        f"{entity}/{project}",
        **query_params
    )
    
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
            "tags": run.tags,
        }
        runs_data.append(run_dict)
    
    return runs_data

@mcp.tool()
def get_run_config(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """
    Fetch configuration parameters for a specific run.
    
    Args:
        entity (str): The wandb entity (username or team name)
        project (str): The project name
        run_id (str): The ID of the run to fetch config for
        
    Returns:
        Dict[str, Any]: Dictionary containing configuration parameters
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    return run.config

@mcp.tool()
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

@mcp.tool()
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

@mcp.tool()
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

if __name__ == "__main__":
    # Start the MCP server
    mcp.run() 