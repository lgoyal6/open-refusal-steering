from public_lab.run import run
import pytest,ast,re,subprocess
from pathlib import Path
def test_shared_classifier_matches_original_functions():
    root=Path(__file__).resolve().parents[1]
    original=subprocess.check_output(['git','show','509c101:src/metrics.py'],cwd=root,text=True)
    tree=ast.parse(original)
    nodes=[n for n in tree.body if isinstance(n,(ast.Assign,ast.FunctionDef)) and not getattr(n,'decorator_list',[])]
    ns={"re":re};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original>','exec'),ns)
    for text in ['', 'a'*40, "I'm sorry, but I cannot help.", 'word '*30,'A sorting algorithm puts items in order.']:
        assert run({"text":text})['classification']==ns['classify'](text)
    assert run({"text":'A sorting algorithm puts items in order.'})['classification']=='ANSWERED'
    assert run({"text":"I'm sorry, but I cannot help."})['classification']=='REFUSAL'
    assert run({"text":"I'm sorry. "*30})['classification']=='DEGENERATE'
    with pytest.raises(ValueError):run({"text":"a"*4097})
