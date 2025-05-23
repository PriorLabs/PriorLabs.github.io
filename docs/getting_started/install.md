**Client** [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KKKOVJk-5N972ZRUeGmRXh8EibRIRGxA#scrollTo=o03aOVAw0Etg)
**Local** [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SHa43VuHASLjevzO7y3-wPCxHY18-2H6#scrollTo=o03aOVAw0Etg&line=3&uniqifier=1)

You can access our models through our API (https://github.com/automl/tabpfn-client), via our user interface built on top of the API (https://www.ux.priorlabs.ai/) or locally.

=== "Python API Client (No GPU, Online)"

    ```bash
    pip install tabpfn-client

    # TabPFN Extensions installs optional functionalities around the TabPFN model
    # These include post-hoc ensembles, interpretability tools, and more
    git clone https://github.com/PriorLabs/tabpfn-extensions
    pip install -e tabpfn-extensions
    ```

=== "Python Local (GPU)"

    ```bash
    # TabPFN Extensions installs optional functionalities around the TabPFN model
    # These include post-hoc ensembles, interpretability tools, and more
    pip install tabpfn
    ```

=== "Web Interface"

    You can access our models through our Interface [here](https://www.ux.priorlabs.ai/).

=== "R"

    !!! warning
        R support is currently under development.
        You can find a work in progress at [TabPFN R](https://github.com/robintibor/R-tabpfn).
        Looking for contributors!
