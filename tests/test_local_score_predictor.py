import json
import pytest
from tools.local_score_predictor import aggregate, fit, predict, calibrate


def test_failure_stays_in_population_denominator():
    cases=[{'unit':'a','family':'rates'}, {'unit':'b','family':'rates'}, {'unit':'c','family':'macro'}]
    loss,families=aggregate(cases,[{'unit':'a','loss':0.}, {'unit':'c','loss':1.}],{'rates':90,'macro':10})
    assert families['rates']==4.
    assert loss==3.7
    with pytest.raises(ValueError):aggregate(cases,[{'unit':'a','loss':1.}]*2,{'rates':90,'macro':10})


def test_bad_direction_cannot_become_positive_slope():
    assert fit([1,2,3],[1,2,3])['slope']==0
    assert predict(fit([1,2,3],[-1,-2,-3]),4)==-4
    assert fit([1,1],[-2,-4])=={'intercept':-3.,'slope':0.}


def test_latest_label_cannot_leak_into_latest_prediction(tmp_path):
    roster={'cases':[{'unit':'a','family':'rates','m0_loss':1.}], 'population':{'rates':1}, 'limitations':[]}
    labels = {'v1': -2., 'v2': -3., 'v3': -4.}
    for i,name in enumerate(labels):
        p=tmp_path/'replays'/name/'summary.json';p.parent.mkdir(parents=True)
        p.write_text(json.dumps({'local_loss':i+1.,'failed':0}))
    first=calibrate(tmp_path,roster,labels)['latest_version_holdout']['prediction']
    labels['v3']=-100
    assert calibrate(tmp_path,roster,labels)['latest_version_holdout']['prediction']==first


def test_failed_result_cannot_smuggle_a_good_loss():
    cases=[{'unit':'a','family':'rates'}]
    assert aggregate(cases,[{'unit':'a','status':'failed','loss':0}],{'rates':1})[0]==8


def test_invalid_or_uncovered_population_is_rejected():
    cases=[{'unit':'a','family':'rates'}]
    with pytest.raises(ValueError):aggregate(cases,[{'unit':'a','loss':float('nan')}],{'rates':1})
    with pytest.raises(ValueError):aggregate(cases,[{'unit':'a','loss':1}],{'rates':1,'macro':1})
