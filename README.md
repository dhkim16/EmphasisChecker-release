# EmphasisChecker

This repository includes code for the EmphasisChecker tool presented in *EmphasisChecker: A tool for Guiding Chart and Caption Emphasis* (IEEE VIS 2023 submission).

## Running the Code

The code includes server-side code and client-side code. The server-side code is located inside the `server` directory and the client-side code is located inside the `public` directory.

### Step 0. Dependencies

- Server-side code: The list of dependencies we installed in our virtual environment for running the code is included in `server/requirements.txt`. When you run the code for the first time, run the following command in python.
```python
stanza.install_corenlp()
stanza.download_corenlp_models(model='english-kbp', version='4.5.0') # model='english-kbp'
```
- Client-side code: You do not need to install any dependencies to be able to run the client-side code. The dependencies of the client-side code has already been included in `public/third-party` and includes [jQuery](https://jquery.com/), [d3](https://d3js.org/), [noUiSlider](https://refreshless.com/nouislider/), [jQuery highlightTextarea](https://garysieling.github.io/jquery-highlighttextarea/), [rdp](https://observablehq.com/@chnn/running-ramer-douglas-peucker-on-typed-arrays) (code taken from the example).

### Step 1. Configurations

- Server-side code: You have to first figure out how you plan on hosting your server. Based on the decision, configure the `FLASK_RUN_PORT` variable defined in `server/server.py`. You may also need to make edits on the final line of the file depending on your decision.
- Client-side code: Once you have decided how the server-side code is run, you need to configure the client-side code to be able to communicate with the server. Inside `public/js/options.js`, you should add the url to the server in `urls` and choose the option in `URL_OPTIONS`. You should also add the port number in `PORTNUM`.

### Step 2. Running the code

- Server-side code: Run `server/server.py` using Python3. (e.g., `python server.py`)
- Client-side code: Host the `public` directory. If you wish to run it locally, you could use packages like `http-server`. The `index.html` file includes the webpage on which you can use the EmphasisChecker tool.

#### Notes:
- The EmphasisChecker has been developed based on the Chrome browser and may not function correctly in other browsers.
- We recommend using a computer for using the EmphasisChecker tool. The interface has not been optimized for smaller screens (e.g., mobile devices).
