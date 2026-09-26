from dash import Dash
from BondSwap.riskFree import RiskFree
from BondSwap.bond import Bond
from BondSwap.swap import Swap
from Appli.analyse import Analyse  # Assuming the Analyse class is in a file named Analyse.py

# Initialize the Dash app
app = Dash(__name__)

# Instantiate the Analyse class
analyse = Analyse()

# Set the layout of the app to the layout defined in the Analyse class
app.layout = analyse.layout

# Register the callbacks defined in the Analyse class
analyse.get_callbacks(app)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)