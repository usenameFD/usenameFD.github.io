from dash import dcc, html, Input, Output
import numpy as np
import plotly.graph_objs as go
from BondSwap.riskFree import RiskFree
from BondSwap.bond import Bond
from BondSwap.swap import Swap

api_key = '99b15e0a2f3b3f4571893e831fd555d0'
date = '31-decembre-2024'

class Analyse:
    def __init__(self):
        # Initialize the layout
        self.layout = self._create_layout()

    def _create_layout(self):
        """Create the layout for the Dash app."""
        return html.Div(style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f4f7f6'}, children=[
            html.Div([
                html.H1("Bond and Swap Pricing Tool", style={'textAlign': 'center', 'color': '#2c3e50'}),
                html.P("Interactive tool to visualize bond prices, swap values, and zero-coupon rates", 
                       style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '18px'}),
            ]),

            # Bond Input Section
            html.Div([
                html.Div([
                    html.H2("Bond Pricing", style={'color': '#2c3e50'}),
                    html.Label("Country:"),
                    dcc.Dropdown(
                        id="bond-country",
                        options=[{'label': 'USA', 'value': 'usa'}, {'label': 'France', 'value': 'fr'}],
                        value='usa',
                        style={'width': '100%', 'padding': '10px'}
                    ),
                    html.Label("Face Value:"),
                    dcc.Input(id="bond-face-value", type="number", value=1000, style={'width': '100%', 'padding': '10px'}),
                    html.Label("Coupon Rate (Annual):"),
                    dcc.Input(id="bond-coupon-rate", type="number", value=0.05, style={'width': '100%', 'padding': '10px'}),
                    html.Label("Maturity (Years):"),
                    dcc.Input(id="bond-maturity", type="number", value=5, style={'width': '100%', 'padding': '10px'}),
                    html.Label("Coupon Frequency (Years):"),
                    dcc.Input(id="bond-freq", type="number", value=0.5, style={'width': '100%', 'padding': '10px'}),
                    html.Button('Calculate Bond Price', id='calculate-bond', n_clicks=0, 
                               style={'width': '100%', 'padding': '10px', 'backgroundColor': '#3498db', 'color': 'white', 'border': 'none'}),
                    html.Div(id="bond-price", style={'fontSize': '20px', 'fontWeight': 'bold', 'textAlign': 'center', 'marginTop': '20px'}),
                ], style={'backgroundColor': 'white', 'borderRadius': '10px', 'padding': '20px', 'boxShadow': '0px 4px 6px rgba(0, 0, 0, 0.1)', 'margin': '20px'}),

                # Bond Price vs Maturity Chart
                dcc.Graph(id="bond-price-maturity", style={'height': '400px'})
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}),

            # Swap Input Section
            html.Div([
                html.Div([
                    html.H2("Interest Rate Swap Pricing", style={'color': '#2c3e50'}),
                    html.Label("Country:"),
                    dcc.Dropdown(
                        id="swap-country",
                        options=[{'label': 'USA', 'value': 'usa'}, {'label': 'France', 'value': 'fr'}],
                        value='usa',
                        style={'width': '100%', 'padding': '10px'}
                    ),
                    html.Label("Notional Amount:"),
                    dcc.Input(id="swap-notional", type="number", value=1000000, style={'width': '100%', 'padding': '10px'}),
                    html.Label("Swap Rate (K):"),
                    dcc.Input(id="swap-rate", type="number", value=0.02, style={'width': '100%', 'padding': '10px'}),
                    html.Label("Payment Times (Years, comma-separated):"),
                    dcc.Input(id="swap-times", type="text", value="1,2,3,4,5", style={'width': '100%', 'padding': '10px'}),
                    html.Button('Calculate Swap Value', id='calculate-swap', n_clicks=0, 
                               style={'width': '100%', 'padding': '10px', 'backgroundColor': '#e67e22', 'color': 'white', 'border': 'none'}),
                    html.Div(id="swap-value", style={'fontSize': '20px', 'fontWeight': 'bold', 'textAlign': 'center', 'marginTop': '20px'}),
                ], style={'backgroundColor': 'white', 'borderRadius': '10px', 'padding': '20px', 'boxShadow': '0px 4px 6px rgba(0, 0, 0, 0.1)', 'margin': '20px'}),

                # Swap Value vs Swap Rate Chart
                dcc.Graph(id="swap-rate-chart", style={'height': '400px'})
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}),

            # Zero-Coupon Rates Curve Chart
            html.Div([
                dcc.Graph(id="zero-coupon-rate-curve", style={'height': '400px', 'marginTop': '20px'})
            ], style={'marginTop': '40px'})
        ])

    def get_callbacks(self, app):
        """Define the callbacks for the Dash app."""

        # Callback for Bond Pricing and Maturity Chart
        @app.callback(
            [Output("bond-price", "children"),
             Output("bond-price-maturity", "figure")],
            [Input("calculate-bond", "n_clicks")],
            [Input("bond-country", "value"),
             Input("bond-face-value", "value"),
             Input("bond-coupon-rate", "value"),
             Input("bond-maturity", "value"),
             Input("bond-freq", "value")]
        )
        def calculate_bond(n_clicks, country, face_value, coupon_rate, maturity, freq):
            if n_clicks > 0:
                # Create Bond object
                bond = Bond()
                bond.get_country(country)
                rf = RiskFree()
                rf.get_country(country)
                rf.get_riskFree_usa(api_key) if country == 'usa' else rf.get_riskFree_fr(date)
                bond.get_riskFree_rate(rf)

                # Calculate Bond Price
                price = bond.get_price(face_value, coupon_rate, maturity, freq)

                # Generate Bond Price vs Maturity Chart
                maturities = np.arange(0.5, maturity + 0.5, 0.5)
                prices = [bond.get_price(face_value, coupon_rate, m, freq) for m in maturities]

                bond_price_maturity = {
                    'data': [go.Scatter(x=maturities, y=prices, mode='lines', name='Bond Price', line={'color': '#3498db'})],
                    'layout': go.Layout(
                        title="Bond Price vs Maturity",
                        xaxis={'title': 'Maturity (Years)', 'showgrid': False, 'zeroline': False},
                        yaxis={'title': 'Bond Price', 'showgrid': False, 'zeroline': False},
                        plot_bgcolor='#ecf0f1'
                    )
                }

                return f"Bond Price: {price:.2f}", bond_price_maturity
            return "", {}

        # Callback for Swap Value and Swap Rate Chart
        @app.callback(
            [Output("swap-value", "children"),
             Output("swap-rate-chart", "figure")],
            [Input("calculate-swap", "n_clicks")],
            [Input("swap-country", "value"),
             Input("swap-notional", "value"),
             Input("swap-rate", "value"),
             Input("swap-times", "value")]
        )
        def calculate_swap(n_clicks, country, notional, rate, times):
            if n_clicks > 0:
                # Create Swap object
                swap = Swap()
                swap.get_country(country)
                rf = RiskFree()
                rf.get_country(country)
                rf.get_riskFree_usa(api_key) if country == 'usa' else rf.get_riskFree_fr(date)
                swap.get_riskFree_rate(rf)

                # Convert times to a list of floats
                T = list(map(float, times.split(',')))
                swap_value = swap.swap(0, T, notional, rate)

                # Generate Swap Value vs Swap Rate Chart
                rates = np.linspace(0.01, 0.1, 20)
                swap_values = [swap.swap(0, T, notional, r) for r in rates]

                swap_rate_chart = {
                    'data': [go.Scatter(x=rates, y=swap_values, mode='lines', name='Swap Value', line={'color': '#e67e22'})],
                    'layout': go.Layout(
                        title="Swap Value vs Swap Rate",
                        xaxis={'title': 'Swap Rate (K)', 'showgrid': False, 'zeroline': False},
                        yaxis={'title': 'Swap Value', 'showgrid': False, 'zeroline': False},
                        plot_bgcolor='#ecf0f1'
                    )
                }

                return f"Swap Value: {swap_value:.2f}", swap_rate_chart
            return "", {}

        # Callback for Zero-Coupon Rates Curve
        @app.callback(
            Output("zero-coupon-rate-curve", "figure"),
            [Input("bond-country", "value")]
        )
        def plot_zero_coupon_rate_curve(country):
            # Create a RiskFree object to fetch the zero-coupon rates
            rf = RiskFree()
            rf.get_country(country)
            rf.get_riskFree_usa(api_key) if country == 'usa' else rf.get_riskFree_fr(date)

            # Simulate maturities and calculate zero-coupon rates
            maturities = np.arange(0.5, 30.5, 0.5)  # Maturities from 0.5 to 30 years
            zero_coupon_rates = rf.riskFree_rate(maturities)

            # Generate Zero-Coupon Rate Curve
            zero_coupon_curve = {
                'data': [go.Scatter(x=maturities, y=zero_coupon_rates, mode='lines', name='Zero-Coupon Rate', line={'color': '#2ecc71'})],
                'layout': go.Layout(
                    title="Zero-Coupon Rate Curve",
                    xaxis={'title': 'Maturity (Years)', 'showgrid': False, 'zeroline': False},
                    yaxis={'title': 'Zero-Coupon Rate (%)', 'showgrid': False, 'zeroline': False},
                    plot_bgcolor='#ecf0f1'
                )
            }

            return zero_coupon_curve