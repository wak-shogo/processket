import streamlit as st
import optuna
import pandas as pd
from optuna.terminator import report_cross_validation_scores
from optuna.visualization import plot_terminator_improvement

st.title("Processket")

st.write("Suggest parameter for next experiment based on past results")

st.markdown("#### Upload data")

uploaded_file = st.file_uploader("",type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    print(df)
    io_row = df.iloc[0]
    
    config_row = df.iloc[1]
    print(config_row)
    config_dict = {}
    for i, v in config_row.items():
        if v == "c":
            config_dict[i] = "continuous"
        elif v == "d":
            config_dict[i] = "discrete"
            
    limit_row = df.iloc[2]
    
    limit_dict = {}
    for i, v in limit_row.items():
        if i in config_dict:
            limit_dict[i] = list(map(float, v.replace("[","").replace("]","").split(",")))
        
    df = df.iloc[3:]
    
    input_columns = [col for col in df.columns if io_row[col] == "i"]
    output_columns = [col for col in df.columns if io_row[col] == "o"]
    
    st.dataframe(pd.concat([df[input_columns],df[output_columns]],axis=1))
    
    study = optuna.create_study(direction="minimize")
    for index, row in df.iterrows():
        params = {key: row[key] for key in input_columns}
        value = row[output_columns[0]]
        distributions = {key: optuna.distributions.FloatDistribution(limit_dict[key][0],limit_dict[key][1])
                         if config_dict[key] == "continuous"
                         else optuna.distributions.CategoricalDistribution(limit_dict[key])
                         for key in input_columns}
        trial = optuna.trial.create_trial(params=params, distributions=distributions, value=value)
        study.add_trial(trial)
        
    def objective(trial):
        params = {key: trial.suggest_float(key, limit_dict[key][0], limit_dict[key][1])
                  if config_dict[key] == "continuous"
                  else trial.suggest_categorical(key, limit_dict[key])
                  for key in config_dict.keys()}
        return 0
    
    study.optimize(objective, n_trials=1)
    
    st.write("next parameters:")
    st.write(study.trials[-1].params)
    
    #visualize optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    st.plotly_chart(fig)
    #visualize contour
    fig = optuna.visualization.plot_contour(study, params=input_columns)
    st.plotly_chart(fig)
    #visualize parameter importance
    fig = optuna.visualization.plot_param_importances(study)
    st.plotly_chart(fig)
    #visualize terminator improvement
    fig = plot_terminator_improvement(study, plot_error=False)
    st.plotly_chart(fig)