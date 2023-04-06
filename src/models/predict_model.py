import pandas as pd

#prediction for list of input keywords
def run_predict_for_series(model,series_kw):
    list_of_series = []
    for selected_keyword in series_kw:
        list_of_k = [k for k in range(1,3)]
        predictions = [prediction(model,selected_keyword,k=i) for i in list_of_k]
        predictions = pd.Series(predictions, index=list_of_k, name=selected_keyword)
        list_of_series.append(predictions)

    result = pd.concat(list_of_series, axis=1).T
    return result

# function to run predictions for a single kw and 
# make list of labels 
def prediction(model, keyword,k=1):
    x=model.predict(keyword,k=k)
    list_label=[]
    for k in range(1,k+1):
        pred_label=(x[0][k-1].split('__')[2])
        list_label.append(pred_label)
        tuple_label=tuple(list_label)
    return tuple_label


