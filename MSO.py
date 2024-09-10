import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
import ast
import psycopg2
from scipy.optimize import minimize, curve_fit, Bounds
import logging

# Set logging logs
logging.basicConfig(level=logging.INFO,  # Nivel de log
                    format='%(asctime)s - %(levelname)s: %(message)s',  # Formato de salida
                    datefmt='%Y-%m-%d %H:%M:%S')

def MercurySearchOptimization(date: str,
                              advertiser: str,
                              period: float,
                              budget: float,
                              target: str,
                              budget_allocation_constraints: tuple,
                              development: bool=False):
    logging.info(f"Starting the execution for the advertiser {advertiser} with target = {target}. Execution date: {date}")
    mso_optimization = MSO(date=date, advertiser=advertiser, target=target, period=period, budget=budget, budget_allocation_constraints=budget_allocation_constraints, development=development)
    logging.info(f"Extracting data from database for advertiser: {advertiser}")
    mso_optimization.get_data_from_db()
    logging.info("Fitting cost curves for each search engine.")
    mso_optimization.get_parameters_from_fit()
    logging.info("Optimising budget allocation.")
    result = mso_optimization.optimize_budget_allocation()
    return result

class MSO:
    def __init__(self,
                 date: str,
                 advertiser: str,
                 period: float,
                 budget: float,
                 budget_allocation_constraints: tuple,
                 target: str,
                 development: bool=False):
        self.conn, self.engine = self.connection(development=development)
        self.date = date
        self.advertiser = advertiser
        self.analysis_id = advertiser + "_" + date
        self.category = self.get_category_from_advertiser()
        self.budget = budget
        self.period = period
        self.budget_allocation_constraints = budget_allocation_constraints
        if target.lower() in ["consideration", "performance"]:
            self.target = target
        else:
            raise Exception[f"The paramenter target = {target} is not valid. Please provide a valid target."]
    
    def connection(self,
                   development: bool=False):
        if development:
            logging.info('Connecting to the pre-mso-db database.')
            conn = psycopg2.connect(host="34.175.36.220", database="preMSO", user ="postgres", password="M9uarT7-1q0ts|l6")   
            engine = create_engine("postgresql+psycopg2://{user}:{password}@{host}/{dbname}".format(
                user="postgres",
                password="M9uarT7-1q0ts|l6",
                host="34.175.36.220",
                dbname="preMSO"))
        else:
            logging.info('Connecting to the pro-mso-db database.')
        #todo: crear entorno pro
        return conn, engine
    
    def get_category_from_advertiser(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT category FROM d_advertisers where advertiser = '" + self.advertiser + "';")
        category = cursor.fetchall()
        cursor.close()
        return category[0][0]
    
    def get_weights_from_db(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM f_index_weights where target = 'consideration';")
        consideration_weights = cursor.fetchall()[0][0]
        cursor.close()
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM f_index_weights where target = 'performance';")
        performance_weights = cursor.fetchall()[0][0]
        cursor.close()
        return consideration_weights/np.sum(consideration_weights), performance_weights/np.sum(performance_weights)
    
    def get_features_matrix_from_db(self):
        # Get f_search_engine_features table from db
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * FROM f_search_engine_features WHERE category = '{self.category}';")
        column_names = [desc[0] for desc in cursor.description]
        f_search_engine_features_table = pd.DataFrame(cursor.fetchall(), columns=column_names)
        cursor.close()
        features_matrix_df = pd.DataFrame(columns=["google", "bing", "meta", "tiktok", "pinterest", "x", "amazon", "youtube"])
        for search_engine in features_matrix_df.columns:
            features_matrix_df[search_engine] = f_search_engine_features_table[f_search_engine_features_table["search_engine"]==search_engine]["value"].values
        return features_matrix_df.div(features_matrix_df.sum(axis=1), axis=0)

    def get_budget_limits_allocation_from_db(self):
        # Get table f_budget_allocation_constraints from db
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * FROM f_budget_allocation_constraints WHERE analysis_id = '{self.analysis_id}' AND target = '{self.target}';")
        column_names = [desc[0] for desc in cursor.description]
        f_budget_allocation_constraints_table = pd.DataFrame(cursor.fetchall(), columns=column_names)
        cursor.close()
        min_limits = f_budget_allocation_constraints_table["lower_bound"].values
        max_limits = f_budget_allocation_constraints_table["upper_bound"].values
        return (min_limits, max_limits)

    def get_data_from_db(self):
        self.weights = self.get_weights_from_db()
        self.features_matrix = self.get_features_matrix_from_db()
        # self.budget_allocation_constraints = self.get_budget_limits_allocation_from_db()
        
    def function_definition(self,
                            x: np.ndarray,
                            k: float,
                            a: float):
        return a*x**k

    def fit_curve(self,
                  x: np.ndarray,
                  y: np.ndarray):
        popt, _ = curve_fit(lambda x, k, a: self.function_definition(x, k, a), x, y, p0=[1, 0.5], bounds=([0, 0], [1, np.inf]), maxfev=10000)
        return popt[0], popt[1]

    def get_parameters_from_fit(self):
        # Get table f_cost_curves from db
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * FROM f_cost_curves WHERE advertiser = '{self.advertiser}';")
        column_names = [desc[0] for desc in cursor.description]
        f_cost_curves_table = pd.DataFrame(cursor.fetchall(), columns=column_names)
        cursor.close()
        search_engines = f_cost_curves_table['search_engine'].unique()
        self.search_engines = search_engines
        k_impressions_list = []
        a_impressions_list = []
        k_clicks_list = []
        a_clicks_list = []
        for search_engine in search_engines:
            # Get data for every search engine
            search_engine_data = f_cost_curves_table[f_cost_curves_table['search_engine'] == search_engine]
            # Insert (0,0) point in all graphs (not needed)
            investments = np.insert(np.array(ast.literal_eval(search_engine_data['investment'].values[0])), 0, 0)
            impressions = np.insert(np.array(ast.literal_eval(search_engine_data[search_engine_data["metric"]=="impressions"]['values'].values[0])), 0, 0)
            clicks = np.insert(np.array(ast.literal_eval(search_engine_data[search_engine_data["metric"]=="clicks"]['values'].values[0])), 0, 0)
            # Fit curves to data
            k_impressions, a_impressions = self.fit_curve(x=investments, y=impressions)
            k_clicks, a_clicks = self.fit_curve(x=investments, y=clicks)
            k_impressions_list.append(k_impressions)
            a_impressions_list.append(a_impressions)
            k_clicks_list.append(k_clicks)
            a_clicks_list.append(a_clicks)
        self.curves_parameters_dict = ({"k": k_impressions_list, "a": a_impressions_list}, {"k": k_clicks_list, "a": a_clicks_list})

    def objective_function_copy(self,
                           x: np.ndarray):
        # Get the parameters from the curves and the weights for the index
        if self.target == "consideration":
            k = self.curves_parameters_dict[0]["k"]
            a = self.curves_parameters_dict[0]["a"]
            weights = self.weights[0]
        else:
            k = self.curves_parameters_dict[1]["k"]
            a = self.curves_parameters_dict[1]["a"]
            weights = self.weights[1]
            # Calculate the index value
        index_contribution = 0.0
        for i, x_i in enumerate(x):
            inv = x_i * self.budget/self.period
            pressure = self.function_definition(x=inv, k=k[i], a=a[i])
            index_contribution = index_contribution + np.sum(weights * self.features_matrix.values[:,i]) * pressure
        return index_contribution

    def get_investment(self,
                       ind: int,
                       x: float):
        inv = x * self.budget/self.period
        if ind == 2 and inv < 5000/self.period:
            inv = 0.0
        elif ind == 5 and inv < 3000/self.period:
            inv = 0.0
        elif ind == 1 and inv < 4000/self.period:
            inv = 0.0
        return inv

    def objective_function(self,
                           x: np.ndarray):
        # Get the parameters from the curves and the weights for the index
        if self.target == "brand":
            k = self.curves_parameters_dict[0]["k"]
            a = self.curves_parameters_dict[0]["a"]
            weights = self.weights[0]
        else:
            k = self.curves_parameters_dict[1]["k"]
            a = self.curves_parameters_dict[1]["a"]
            weights = self.weights[1]
            # Calculate the index value
        index_contribution = 0.0
        for i, x_i in enumerate(x):
            inv = x_i * self.budget/self.period
            pressure = self.function_definition(x=inv, k=k[i], a=a[i])
            index_contribution = index_contribution + np.sum(weights * self.features_matrix.values[:,i]) * pressure
        return -index_contribution/self.initial_guess
    
    def budget_constraint(self,
                          x: np.ndarray):
        return 1 - np.sum(x)
    
    def adapt_solution_to_constraints(self,
                                      solution: np.ndarray):
        inv = solution * self.budget/self.period
        if inv[2] < 5000/self.period:
            solution[2] = 0.0
        elif inv[5] < 3000/self.period:
            solution[5] = 0.0
        elif inv[1] < 4000/self.period:
            solution[1] = 0.0
        return solution/np.sum(solution)

    def update_table_by_field(self,
                              tablename: str,
                              df: pd.DataFrame,
                              field: str,
                              field_value: str):
        try:
            # Delete rows with the same category
            cursor = self.conn.cursor()
            query=f"DELETE FROM {tablename} WHERE {field}  = '{field_value}';"
            cursor.execute(query)
            self.conn.commit()
            cursor.close()
            # Load new data
            df.to_sql(tablename, con=self.engine,if_exists='append',index=False)
        except:
            # Load new data
            df.to_sql(tablename, con=self.engine, if_exists='append',index=False)

    def upload_results_to_db(self,
                             solution: np.ndarray):
        logging.info("Uploading results to mso-db.")
        f_budget_allocation_results_df = pd.DataFrame(columns=["analysis_id", "search_engine", "target", "budget_pct"])
        f_budget_allocation_results_df["search_engine"] = self.search_engines
        f_budget_allocation_results_df["target"] = self.target
        f_budget_allocation_results_df["budget_pct"] = solution
        f_budget_allocation_results_df["analysis_id"] = self.analysis_id
        self.update_table_by_field(tablename="f_budget_allocation_results",
                                   df=f_budget_allocation_results_df,
                                   field="analysis_id",
                                   field_value=self.analysis_id)
    
    def optimize_budget_allocation(self):
        objective_values = []
        def callback(xk):
            # print results
            obj_value = self.objective_function(xk)
            objective_values.append(obj_value)
            print(f"Iteration {len(objective_values)}: Objective function value = {obj_value}")
            print("x_pct: " + str(xk))
        # Constraints
        constraints = [{"type": "ineq",
                        "fun": self.budget_constraint}]
        # Define starting point
        x0 = (self.budget_allocation_constraints[0] + self.budget_allocation_constraints[1])/2
        x0 = self.budget_allocation_constraints[0]
        x0 = x0/np.sum(x0)
        self.initial_guess = self.objective_function_copy(x0)
        # Reformat the bounds
        bounds = Bounds(lb=self.budget_allocation_constraints[0], ub=self.budget_allocation_constraints[1])
        result = minimize(self.objective_function,
                          x0=x0,
                          bounds=bounds,
                          options={
                              "maxiter": 100,
                              "disp": True,
                              "ftol": 1e-6,
                              "eps": 1.5e-8},
                          constraints=constraints,
                          callback=callback)
        solution = self.adapt_solution_to_constraints(solution=result.x)
        self.upload_results_to_db(solution=solution)
        return result

