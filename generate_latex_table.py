import argparse
import pickle
from collections import defaultdict
from statistics import mean
from scipy.stats import wilcoxon
from numpy import std
from run_experiments import factory, to_dict

min_metrics = ['ece', 'brier']

renames = {
    'bn_realistic' : 'BN-only',
    'gt_bn' : 'GT-only',
    'binary_classifiers' : 'text-only',
    'tabular_text_binary' : 'Concat-text-tab',
    'binary_classifiers_data_shift' : 'text-only',
    'weighted_consistency' : 'C-BN-text',
    'virtual' : 'V-BN-text',
    'weighted_consistency_virtual' : 'V-C-BN-text',
    'weighted_consistency_ground_truth' : 'C-BN-text',
    'virtual_ground_truth' : 'V-BN-text',
    'weighted_consistency_virtual_ground_truth' : 'V-C-BN-text',
    'weighted_consistency_data_shift' : 'C-BN-text',
    'virtual_data_shift' : 'V-BN-text',
    'weighted_consistency_virtual_data_shift' : 'V-C-BN-text'
}

model_order = list(renames.keys())

baseline_models = ['bn_realistic', 'gt_bn', 'binary_classifiers', 'tabular_text_binary', 'binary_classifiers_data_shift']

def keep_model(model: str) -> bool:
    return (keep_models == 'all' or model in keep_models) and model in renames

def tabstring(s: str, i=1) -> str:
    return '  '*i + s

def display_num(num: float, decimals:int=4) -> str:
    return str(round(num, decimals))

def dwim_name(name: str | int) -> str:
    if name in renames:
        return renames[name]
    else:
        return str(name).replace('_', '-')

def dwim_column_name(name: str | int) -> str:
    dwimmed = dwim_name(name)
    if dwimmed == 'precision':
        return 'p'
    elif dwimmed == 'average-precision':
        return 'ap'
    else:
        return dwimmed
    
class TableEntry():
    def __init__(self, val, is_max, is_significant, is_max_baseline):
        self.val = val
        self.is_max = is_max
        self.is_significant = is_significant
        self.is_max_baseline = is_max_baseline
        self.sd = None
    def __str__(self):
        return f"[Table Entry: {self.val}, {self.is_max}, {self.is_significant}, {self.is_max_baseline}]"
    def __repr__(self):
        return self.__str__()
    def display_value(self) -> str:
        if self.sd is not None:
            val = display_num(self.val*100, 2)
            sd = display_num(self.sd*100, 2)
            dv = f'${val}\\pm{sd}$'
        else:
            dv = display_num(self.val)
            if self.is_significant:
                dv = '*' + dv
            if self.is_max_baseline:
                dv = '\\underline{' + dv +'}'
            if self.is_max:
                dv = '\\textbf{' + dv +'}'
        return dv
    
def metric_to_minimizeP(column, metric):
    return column in min_metrics or metric in min_metrics

def determine_max_models(reaggregated_results: dict, metric: str | None, models=None):
    if models is None:
        models = reaggregated_results.keys()
    else:
        models = [m for m in models if m in reaggregated_results.keys()] # filter to applicable baselines
    max_models = {}
    for model in models:
        for column, vals in reaggregated_results[model].items():
            if column in max_models:
                val = mean(vals)
                if metric_to_minimizeP(column, metric):
                    if val < max_models[column][1]:
                        max_models[column] = (model, val)
                else:
                    if val > max_models[column][1]:
                        max_models[column] = (model, val)
            else:
                max_models[column] = (model, mean(vals))
    max_models = { column: entry[0] for column, entry in max_models.items() }
    return max_models

def interesting_model(result: dict, model: str) -> bool:
    if model in baseline_models:
        return True
    for entry in result[model].values():
        if entry.is_max or entry.is_significant:
            return True
    return False

def calculate_symptom_means(data: dict) -> dict:
    """
    data: a dict of form metric -> symptom -> n_samples -> seed -> model
    Add a symptom 'mean' that averages performance across all symptoms for each metric.
    """
    for metric_data in data.values():
        mean_dict = defaultdict(factory(2))
        n = len(metric_data)
        for symptom_data in metric_data.values():
            for n_samples, sample_data in symptom_data.items():
                for seed, seed_data in sample_data.items():
                    for model, val in seed_data.items():
                        mean_dict[n_samples][seed][model] += val/n
        metric_data['mean'] = to_dict(mean_dict)
    return data

def analyze_symptom_data(data: dict, symptom: str, mode: str, n_samples: int | None = None, metric: str | None = None) -> dict:
    """
    data: a dict of form n_samples -> seed -> model iff mode == 'n_samples'
          a dict of form metric -> symptom -> n_samples -> seed -> model iff mode == 'metric'
    Return (a dict of the form model -> n_samples -> TableEntry iff mode == 'n_samples' OR
            a dict of the form model -> metric -> TableEntry iff mode == 'metric') AND
           a dict of the form column -> change of best non-baseline model over max baseline
    """
    def val_above_baseline(val, column, max_baseline_vals):
        if metric_to_minimizeP(column, metric):
            return val < mean(max_baseline_vals)
        else:
            return val > mean(max_baseline_vals)
    # reaggregate to model -> n_samples -> list of vals
    reaggregated_results = defaultdict(lambda: defaultdict(list))
    if mode == 'n_samples':
        for n_samples, sample_data in data.items():
            for _, seed_data in sample_data.items():
                for model, val in seed_data.items():
                    if keep_model(model):
                        reaggregated_results[model][n_samples].append(val)
    elif mode == 'metric':
        for met, metric_data in data.items():
            for _, seed_data in metric_data[symptom][n_samples].items():
                for model, val in seed_data.items():
                    if keep_model(model):
                        reaggregated_results[model][met].append(val)
    # determine max baseline models
    max_baseline_models = determine_max_models(reaggregated_results, metric, baseline_models)
    # determine max across all models
    max_models = determine_max_models(reaggregated_results, metric)

    # generate table entries
    result = defaultdict(dict)
    num_baselines = len([model for model in reaggregated_results.keys() if model in baseline_models])
    for model, model_data in reaggregated_results.items():
        for column, vals in model_data.items():
            val = mean(vals)
            if len(reaggregated_results) == 1:
                result[model][column] = TableEntry(val, False, False, False)
            else:
                is_max = model == max_models[column]
                max_baseline = max_baseline_models[column]
                max_baseline_vals = reaggregated_results[max_baseline][column]
                is_significant = False
                if val_above_baseline(val, column, max_baseline_vals):
                    alternative = 'greater' if not metric_to_minimizeP(column, metric) else 'less'
                    _, p_value = wilcoxon(vals, max_baseline_vals, alternative=alternative)
                    if p_value < 0.05:
                        is_significant = True
                is_max_baseline = (model == max_baseline and num_baselines > 1)
                if len(reaggregated_results) == 2:
                    result[model][column] = TableEntry(val, False, is_significant, False)
                else:
                    result[model][column] = TableEntry(val, is_max, is_significant, is_max_baseline)
            if standard_deviation:
                result[model][column].sd = std(vals, ddof=1) # sample standard deviation
        if filter_uninteresting and not interesting_model(result, model):
            del result[model]
    # determine max non-baseline models
    non_baseline_models = [model for model in reaggregated_results.keys() if model not in baseline_models]
    max_non_baseline_models = determine_max_models(reaggregated_results, metric, non_baseline_models)
    # determine change from baseline
    change_from_baseline_data = {}
    if max_baseline_models and max_non_baseline_models:
        for column, max_baseline_model in max_baseline_models.items():
            max_baseline_val = mean(reaggregated_results[max_baseline_model][column])
            max_non_baseline_model = max_non_baseline_models[column]
            max_non_baseline_val = mean(reaggregated_results[max_non_baseline_model][column])
            change = max_non_baseline_val -  max_baseline_val
            change_from_baseline_data[column] = change

    return dict(result), change_from_baseline_data

def change_from_baseline_string(change_from_baseline_data: dict) -> str:
    """
    change_from_baseline_data: a dict of the form column -> float
    Return a string for the table line quantifying the change of the best model from the highest baseline.
    """
    row_string = '& \\textbf{change vs. baseline}'
    for column in sorted(change_from_baseline_data.keys()):
        val = change_from_baseline_data[column]
        display_val = display_num(val*100, 2)
        if val > 0:
            display_val = '+'+display_val
        row_string += ' & ' + display_val + '\\%'
    row_string += ' \\\\\n'
    return row_string
    
def symptom_subsection(symptom: str, symptom_data: dict, column_mode: str, n_samples: int | None, metric: str | None) -> str:
    analyzed_data, change_from_baseline_data = analyze_symptom_data(symptom_data, symptom, column_mode, n_samples, metric)

    subsection = tabstring('\\midrule\n', 2)  

    first, drew_baseline = True, False
    for model, row in analyzed_data.items():
        if model not in baseline_models and not drew_baseline and len(analyzed_data) > 2:
            subsection += tabstring('\\cline{' + f'{2}-{len(row)+2}' + '}\n', 2)
            drew_baseline = True
        mn = dwim_name(model)
        row_string = tabstring('\\texttt{' + symptom + '} & \\textbf{' + mn + '}', 2) if first else \
            tabstring('& \\textbf{' + mn + '}', 2)
        first = False
        for n_samples in sorted(row.keys()):
            table_entry = row[n_samples]
            row_string += ' & ' + table_entry.display_value()
        row_string += ' \\\\\n'
        subsection += row_string
    if change_from_baseline_data:
        subsection += tabstring('\\cline{' + f'{2}-{len(row)+2}' + '}\n', 2)
        subsection += tabstring(change_from_baseline_string(change_from_baseline_data), 2)

    return subsection

def generate_latex_table(data: dict, table_name: str | int, symptoms: list[str], columns: str, data_shift_experiment: bool, mode: str|None) -> str:
    """
    data: a dict of form symptom -> n_samples -> seed -> model iff columns == 'n_samples'
          a dict of form metric -> symptom -> n_samples -> seed -> model iff columns == 'metric'
    table_name: the metric or n_samples to generate the table for
    symptoms: the symptoms to generate the table for
    columns: the column type (n_samples, metric)
    Return a string that will render a latex table consolidating results on this metric + symptom
    """
    n_columns = len(data[symptoms[0]]) if columns == 'n_samples' else len(data)
    column_range = sorted(data[symptoms[0]].keys()) if columns == 'n_samples' else sorted(data.keys())

    # begin table
    table = '\\begin{table*}[t]\n'
    table += '\\floatconts\n'
    # caption
    name = str(table_name)
    if len(symptoms) == 1:
        name += '_'+symptoms[0]
    if data_shift_experiment:
        name += '_data_shift'
    if mode:
        name += '_'+mode
    table += tabstring(f'{{tab:comparison_models_{name}}}\n')
    table += tabstring('{\\caption{' + dwim_name(name) + '\\\\The best model per training size and per symptom is highlighted in \\textbf{bold}. The best baseline model for each class is \\underline{underlined}. Cases where a model outperforms the best baseline model significantly are indicated by \\textbf{*} ($p < 0.05$ in a one-sided Wilcoxon signed-rank test over 20 seeds).}}\n')
    table += '{\n'
    table += '\\resizebox{\\textwidth}{!}{\n'
    # tabular
    table += '\\begin{tabular}{ll' + 'c'*n_columns + '}\n'
    table += tabstring('\\toprule\n', 2)
    # header
    column_label = 'Training size $n$' if columns == 'n_samples' else 'Metric'
    table += tabstring('& & \\multicolumn{' + str(n_columns) + '}{c}{' + column_label + '} \\\\ \\cmidrule{3-' + str(n_columns+2) + '}\n', 2)
    table += tabstring('&' + ''.join([ ' & \\textbf{' + dwim_column_name(n) + '}' for n in column_range ]) + '\\\\\n', 2)
    table += tabstring('%\\cmidrule{3-' + str(n_columns+2) + '}\n', 2)
    # symptom subsections
    metric = table_name if columns == 'n_samples' else None
    for symptom in symptoms:
        symptom_data = data[symptom] if columns == 'n_samples' else data
        n_samples = table_name if columns == 'metric' else None
        table += symptom_subsection(symptom, symptom_data, columns, n_samples, metric)
    # end tabular
    table += tabstring('\\bottomrule\n', 2)
    table += tabstring('\\bottomrule\n', 2)
    # end table
    table += '\\end{tabular}'
    table += '}\n'
    table += '}\n'
    table += '\\end{table*}'

    return table

def arbitrary_dict_key(dict: dict):
    return next(iter(dict.keys()))

def arbitrary_dict_val(dict: dict):
    return next(iter(dict.values()))

def arbitrary_nested_dict_val(dict: dict, level: int):
    if level == 0:
        return arbitrary_dict_val(dict)
    else:
        return arbitrary_nested_dict_val(arbitrary_dict_val(dict), level-1)

def is_data_shift_experiment(data: dict) -> bool:
    """
    data: a dict of form metric -> symptom -> n_samples -> seed -> model
    Return a bool indicating if the results are for the data shift experiment
    """
    return 'redacted' in arbitrary_dict_key(arbitrary_nested_dict_val(data, 3))

def dwim_data_shift_experiment_name(model: str) -> str:
    return model.replace('_redacted', '')

def dwim_data_shift_experiment_names(data) -> dict:
    """
    data: a dict of form metric -> symptom -> n_samples -> seed -> model
    """
    new_data = defaultdict(factory(4))
    for metric, metric_data in data.items():
        for symptom, symptom_data in metric_data.items():
            for n_samples, n_samples_data in symptom_data.items():
                for seed, seed_data in n_samples_data.items():
                    for model, model_data in seed_data.items():
                        new_model = dwim_data_shift_experiment_name(model)
                        new_data[metric][symptom][n_samples][seed][new_model] = model_data
    return to_dict(new_data)

def sort_models(data: dict) -> dict:
    """
    data: a dict of form metric -> symptom -> n_samples -> seed -> model
    Return data with the models sorted
    """
    sorted_data = defaultdict(factory(4))
    for metric, metric_data in data.items():
        for symptom, symptom_data in metric_data.items():
            for n_samples, n_samples_data in symptom_data.items():
                for seed, seed_data in n_samples_data.items():
                    sorted_data[metric][symptom][n_samples][seed] = dict(sorted(seed_data.items(), key=lambda x: model_order.index(x[0])))
    return to_dict(sorted_data)

def generate_latex_tables(data: dict, ms: list[str] | str, ss: list[str] | str, combine: bool, columns: str, mode: str|None = None) -> str:
    """
    data: a dict of form metric -> symptom -> n_samples -> seed -> model
    Return a string that will render latex tables consolidating results across each metric + symptom
    """
    data_shift_experiment = is_data_shift_experiment(data)
    if data_shift_experiment:
        data = dwim_data_shift_experiment_names(data)
    data = sort_models(data)
    data = calculate_symptom_means(data)
    tables = []
    if columns == 'n_samples':
        for metric, metric_data in data.items():
            if ms == 'all' or metric in ms:
                if combine and (ss == 'all' or len(ss) > 1):
                    symptoms = list(metric_data.keys()) if ss == 'all' else ss
                    tables.append(generate_latex_table(metric_data, metric, symptoms, columns, data_shift_experiment, mode))
                else:
                    for symptom in metric_data.keys():
                        if ss == 'all' or symptom in ss:
                            tables.append(generate_latex_table(metric_data, metric, [symptom], columns, data_shift_experiment, mode))
    elif columns == 'metric':
        metric_data = next(iter(data.values()))
        symptoms = list(metric_data.keys()) if ss == 'all' else ss
        n_samples_range = metric_data[symptoms[0]].keys()
        for n_samples in n_samples_range:
            if combine:
                tables.append(generate_latex_table(data, n_samples, symptoms, columns, data_shift_experiment, mode))
            else:
                for symptom in symptoms:
                    if ss == 'all' or symptom in ss:
                        tables.append(generate_latex_table(data, n_samples, [symptom], columns, data_shift_experiment, mode))
    return '\n\n'.join(tables)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('destination')
    parser.add_argument('--metrics', '-m', nargs='+', type=str, default='all')
    parser.add_argument('--symptoms', '-s', nargs='+', type=str, default='all')
    parser.add_argument('-c', '--combine_symptoms', action='store_true')
    parser.add_argument('--columns', default='n_samples')
    parser.add_argument('-f', '--filter', action='store_true')
    parser.add_argument('--models', nargs='+', type=str, default='all')
    parser.add_argument('--subsets', action='store_true')
    parser.add_argument('-sd', '--standard_deviation', action='store_true')
    args = parser.parse_args()

    filename = args.filename
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    combine = args.combine_symptoms
    columns = args.columns
    filter_uninteresting = args.filter
    keep_models = args.models
    standard_deviation = args.standard_deviation

    if args.subsets:
        tables = '\n\n'.join([generate_latex_tables(mode_data, args.metrics, args.symptoms, combine, columns, mode) for mode, mode_data in data.items()])
    else:
        tables = generate_latex_tables(data, args.metrics, args.symptoms, combine, columns)

    with open(args.destination, 'w') as file:
        file.write(tables)
        