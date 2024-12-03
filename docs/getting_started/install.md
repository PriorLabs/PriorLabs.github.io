You can access our models through our API (https://github.com/automl/tabpfn-client) or via our user interface built on top of the API (https://www.priorlabs.ai/tabpfn-gui).
We will release open weights models soon, currently we are available via api and via our user interface built on top of the API.

=== "Python API Client (No GPU, Online)"

    ```bash
    pip install tabpfn-client
    ```
    
    To log into your account, create an account through our website or through the programmatic interface. Then log into that account by setting the token with
    
    ```python
	from tabpfn_client import config
	
	# Retrieve Token
	with open(config.g_tabpfn_config.user_auth_handler.CACHED_TOKEN_FILE, 'r') as file:
	    token = file.read()
	print(f"TOKEN: {token}")
	from tabpfn_client import config
	
	# Set Token
	service_client = config.ServiceClient()
	config.g_tabpfn_config.user_auth_handler = config.UserAuthenticationClient(service_client=service_client)
	user_auth = config.g_tabpfn_config.user_auth_handler.set_token(token)
    ```

=== "Python Local (GPU)"

    !!! warning
        Not released yet
