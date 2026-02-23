import json
import pandas as pd
from pathlib import Path
import numpy as np
from collections import defaultdict

# Pfade
EVALUATIONS_BASE_DIR = Path("../../llm_as_a_judge/evaluations")
OUTPUT_DIR = Path("../../llm_as_a_judge/results/llm_judge")

# Zwei verschiedene Evaluationsordner
EVALUATION_DIRS = {
    "standard": EVALUATIONS_BASE_DIR,
    "prompt_engineering": Path("../../llm_as_a_judge/evaluations_two_dimensions")
}

# Modellnamen
JUDGE_MODELS = ["deepseek_r1", "gpt_oss_120b", "llama_3.3_70b_instruct"]
INTERPRETATION_MODELS = ["deepseek", "openai", "llama"]

# Neue Kriterien für das Two-Dimensions-Format
CRITERIA = [
    'halluzination_score',
    'tiefe_score',
    'gesamt_score'
]

def create_output_directory():
    """Erstelle Ausgabeordner für Ergebnisse"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_all_evaluations(eval_base_dir):
    """Lade alle Evaluations-JSONs aus einem spezifischen Verzeichnis"""
    evaluations = defaultdict(lambda: defaultdict(list))
    
    for judge_model in JUDGE_MODELS:
        judge_dir = eval_base_dir / judge_model
        if not judge_dir.exists():
            continue
        
        for json_file in judge_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Überspringe fehlerhafte Evaluationen
                if "error" in data:
                    continue
                
                # Extrahiere Interpretations-Modell aus JSON
                if 'interpretations_modell' in data:
                    interp_model = data['interpretations_modell']
                    evaluations[interp_model][judge_model].append(data)
                    
            except Exception as e:
                print(f"Fehler beim Laden von {json_file}: {e}")
                continue
    
    return evaluations

def calculate_statistics(scores_list):
    """Berechne Statistiken für eine Liste von Scores"""
    if not scores_list:
        return {
            'mean': 0,
            'std': 0,
            'min': 0,
            'max': 0,
            'median': 0,
            'count': 0
        }
    
    return {
        'mean': np.mean(scores_list),
        'std': np.std(scores_list),
        'min': np.min(scores_list),
        'max': np.max(scores_list),
        'median': np.median(scores_list),
        'count': len(scores_list)
    }

def aggregate_evaluations(evaluations):
    """Aggregiere Evaluationen für jedes Interpretations-Modell"""
    results = {}
    
    for interp_model in INTERPRETATION_MODELS:
        model_results = {
            'model': interp_model,
            'by_judge': {},
            'overall': {},
            'by_interpretation_number': {}  # Neu: Analyse pro Iterationsnummer
        }
        
        # Sammle Scores für jedes Kriterium pro Judge
        all_scores_by_criterion = defaultdict(list)
        scores_by_iteration = defaultdict(lambda: defaultdict(list))
        
        for judge_model in JUDGE_MODELS:
            judge_scores = defaultdict(list)
            
            if judge_model in evaluations[interp_model]:
                for evaluation in evaluations[interp_model][judge_model]:
                    # Neues Format: "interpretationen" ist eine Liste
                    if 'interpretationen' in evaluation:
                        for interpretation in evaluation['interpretationen']:
                            interp_num = interpretation.get('nummer', None)
                            
                            for criterion in CRITERIA:
                                if criterion in interpretation:
                                    try:
                                        score_value = float(interpretation[criterion])
                                        judge_scores[criterion].append(score_value)
                                        all_scores_by_criterion[criterion].append(score_value)
                                        
                                        # Speichere auch nach Iterationsnummer
                                        if interp_num is not None:
                                            scores_by_iteration[interp_num][criterion].append(score_value)
                                    except (ValueError, TypeError):
                                        continue
            
            # Berechne Statistiken für diesen Judge
            judge_stats = {}
            for criterion, scores in judge_scores.items():
                judge_stats[criterion] = calculate_statistics(scores)
            
            model_results['by_judge'][judge_model] = judge_stats
        
        # Berechne Gesamtstatistiken über alle Judges
        overall_stats = {}
        for criterion, scores in all_scores_by_criterion.items():
            overall_stats[criterion] = calculate_statistics(scores)
        
        model_results['overall'] = overall_stats
        
        # Berechne Statistiken pro Iterationsnummer
        iteration_stats = {}
        for iteration_num, criterion_scores in scores_by_iteration.items():
            iteration_stats[iteration_num] = {}
            for criterion, scores in criterion_scores.items():
                iteration_stats[iteration_num][criterion] = calculate_statistics(scores)
        
        model_results['by_interpretation_number'] = iteration_stats
        
        results[interp_model] = model_results
    
    return results

def create_summary_dataframe(results):
    """Erstelle DataFrame mit Zusammenfassung"""
    summary_data = []
    
    for interp_model, model_results in results.items():
        row = {'model': interp_model}
        
        # Füge Overall-Means hinzu
        for criterion, stats in model_results['overall'].items():
            row[f'{criterion}_mean'] = round(stats['mean'], 2)
            row[f'{criterion}_std'] = round(stats['std'], 2)
            row[f'{criterion}_count'] = stats['count']
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def create_detailed_dataframe(results):
    """Erstelle detaillierte DataFrame mit Judge-spezifischen Scores"""
    detailed_data = []
    
    for interp_model, model_results in results.items():
        for judge_model, judge_stats in model_results['by_judge'].items():
            for criterion, stats in judge_stats.items():
                detailed_data.append({
                    'interpretation_model': interp_model,
                    'judge_model': judge_model,
                    'criterion': criterion,
                    'mean': round(stats['mean'], 2),
                    'std': round(stats['std'], 2),
                    'min': stats['min'],
                    'max': stats['max'],
                    'median': round(stats['median'], 2),
                    'count': stats['count']
                })
    
    return pd.DataFrame(detailed_data)

def create_iteration_analysis_dataframe(results):
    """Erstelle DataFrame zur Analyse pro Iterationsnummer (1-10)"""
    iteration_data = []
    
    for interp_model, model_results in results.items():
        for iteration_num, criterion_stats in model_results['by_interpretation_number'].items():
            row = {
                'interpretation_model': interp_model,
                'iteration_number': iteration_num
            }
            
            for criterion, stats in criterion_stats.items():
                row[f'{criterion}_mean'] = round(stats['mean'], 2)
                row[f'{criterion}_count'] = stats['count']
            
            iteration_data.append(row)
    
    df = pd.DataFrame(iteration_data)
    
    # Sortiere nach Modell und Iteration
    if not df.empty:
        df = df.sort_values(['interpretation_model', 'iteration_number'])
    
    return df

def create_comparison_table(results):
    """Erstelle Vergleichstabelle der Modelle"""
    comparison_data = []
    for criterion in CRITERIA:
        row = {'criterion': criterion}
        for interp_model in INTERPRETATION_MODELS:
            if criterion in results[interp_model]['overall']:
                row[interp_model] = round(results[interp_model]['overall'][criterion]['mean'], 2)
            else:
                row[interp_model] = None
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def analyze_judge_agreement(results):
    """Analysiere die Übereinstimmung zwischen Judges"""
    agreement_data = []
    
    for interp_model in INTERPRETATION_MODELS:
        for criterion in CRITERIA:
            judge_means = []
            for judge_model in JUDGE_MODELS:
                if judge_model in results[interp_model]['by_judge']:
                    if criterion in results[interp_model]['by_judge'][judge_model]:
                        mean = results[interp_model]['by_judge'][judge_model][criterion]['mean']
                        judge_means.append(mean)
            
            if len(judge_means) > 1:
                agreement_data.append({
                    'interpretation_model': interp_model,
                    'criterion': criterion,
                    'judge_agreement_std': round(np.std(judge_means), 2),
                    'judge_agreement_range': round(max(judge_means) - min(judge_means), 2),
                    'n_judges': len(judge_means)
                })
    
    return pd.DataFrame(agreement_data)

def create_approach_comparison(results_standard, results_prompt_eng):
    """Erstelle Vergleichstabelle zwischen Standard und Prompt Engineering"""
    comparison_data = []
    
    for interp_model in INTERPRETATION_MODELS:
        for criterion in CRITERIA:
            if (criterion in results_standard[interp_model]['overall'] and 
                criterion in results_prompt_eng[interp_model]['overall']):
                
                standard_mean = results_standard[interp_model]['overall'][criterion]['mean']
                prompt_eng_mean = results_prompt_eng[interp_model]['overall'][criterion]['mean']
                difference = prompt_eng_mean - standard_mean
                
                comparison_data.append({
                    'model': interp_model,
                    'criterion': criterion,
                    'standard_mean': round(standard_mean, 2),
                    'prompt_eng_mean': round(prompt_eng_mean, 2),
                    'difference': round(difference, 2),
                    'improvement_percent': round((difference / standard_mean * 100) if standard_mean > 0 else 0, 1)
                })
    
    return pd.DataFrame(comparison_data)

def create_side_by_side_comparison(results_standard, results_prompt_eng):
    """Erstelle Side-by-Side Vergleich für bessere Übersicht"""
    comparison_data = []
    
    for criterion in CRITERIA:
        row = {'criterion': criterion}
        
        for interp_model in INTERPRETATION_MODELS:
            if (criterion in results_standard[interp_model]['overall'] and 
                criterion in results_prompt_eng[interp_model]['overall']):
                
                standard = results_standard[interp_model]['overall'][criterion]['mean']
                prompt_eng = results_prompt_eng[interp_model]['overall'][criterion]['mean']
                
                row[f'{interp_model}_standard'] = round(standard, 2)
                row[f'{interp_model}_prompt_eng'] = round(prompt_eng, 2)
                row[f'{interp_model}_diff'] = round(prompt_eng - standard, 2)
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def save_results(results_standard, results_prompt_eng, summary_df_std, summary_df_pe, 
                detailed_df_std, detailed_df_pe, comparison_df_std, comparison_df_pe, 
                agreement_df_std, agreement_df_pe, approach_comp_df, side_by_side_df,
                iteration_df_std, iteration_df_pe):
    """Speichere alle Ergebnisse"""
    # Speichere vollständige Ergebnisse als JSON
    with open(OUTPUT_DIR / 'full_results_standard.json', 'w', encoding='utf-8') as f:
        json.dump(results_standard, f, ensure_ascii=False, indent=2, default=str)
    
    with open(OUTPUT_DIR / 'full_results_prompt_engineering.json', 'w', encoding='utf-8') as f:
        json.dump(results_prompt_eng, f, ensure_ascii=False, indent=2, default=str)
    
    # Speichere DataFrames als CSV
    summary_df_std.to_csv(OUTPUT_DIR / 'summary_standard.csv', index=False)
    summary_df_pe.to_csv(OUTPUT_DIR / 'summary_prompt_engineering.csv', index=False)
    detailed_df_std.to_csv(OUTPUT_DIR / 'detailed_scores_standard.csv', index=False)
    detailed_df_pe.to_csv(OUTPUT_DIR / 'detailed_scores_prompt_engineering.csv', index=False)
    comparison_df_std.to_csv(OUTPUT_DIR / 'model_comparison_standard.csv', index=False)
    comparison_df_pe.to_csv(OUTPUT_DIR / 'model_comparison_prompt_engineering.csv', index=False)
    agreement_df_std.to_csv(OUTPUT_DIR / 'judge_agreement_standard.csv', index=False)
    agreement_df_pe.to_csv(OUTPUT_DIR / 'judge_agreement_prompt_engineering.csv', index=False)
    approach_comp_df.to_csv(OUTPUT_DIR / 'approach_comparison.csv', index=False)
    side_by_side_df.to_csv(OUTPUT_DIR / 'side_by_side_comparison.csv', index=False)
    iteration_df_std.to_csv(OUTPUT_DIR / 'iteration_analysis_standard.csv', index=False)
    iteration_df_pe.to_csv(OUTPUT_DIR / 'iteration_analysis_prompt_engineering.csv', index=False)
    
    # Speichere auch als Excel für bessere Lesbarkeit
    with pd.ExcelWriter(OUTPUT_DIR / 'evaluation_results_complete.xlsx', engine='openpyxl') as writer:
        # Standard Approach
        summary_df_std.to_excel(writer, sheet_name='Summary_Standard', index=False)
        detailed_df_std.to_excel(writer, sheet_name='Detailed_Standard', index=False)
        comparison_df_std.to_excel(writer, sheet_name='Comparison_Standard', index=False)
        agreement_df_std.to_excel(writer, sheet_name='Agreement_Standard', index=False)
        iteration_df_std.to_excel(writer, sheet_name='Iteration_Standard', index=False)
        
        # Prompt Engineering Approach
        summary_df_pe.to_excel(writer, sheet_name='Summary_PromptEng', index=False)
        detailed_df_pe.to_excel(writer, sheet_name='Detailed_PromptEng', index=False)
        comparison_df_pe.to_excel(writer, sheet_name='Comparison_PromptEng', index=False)
        agreement_df_pe.to_excel(writer, sheet_name='Agreement_PromptEng', index=False)
        iteration_df_pe.to_excel(writer, sheet_name='Iteration_PromptEng', index=False)
        
        # Approach Comparison
        approach_comp_df.to_excel(writer, sheet_name='Approach_Comparison', index=False)
        side_by_side_df.to_excel(writer, sheet_name='Side_by_Side', index=False)

def print_summary(comparison_df_std, comparison_df_pe, side_by_side_df, iteration_df_std, iteration_df_pe):
    """Drucke Zusammenfassung in Konsole"""
    print("\n" + "="*70)
    print("STANDARD APPROACH - DURCHSCHNITTLICHE SCORES")
    print("="*70)
    print(comparison_df_std.to_string(index=False))
    print("="*70 + "\n")
    
    print("\n" + "="*70)
    print("PROMPT ENGINEERING APPROACH - DURCHSCHNITTLICHE SCORES")
    print("="*70)
    print(comparison_df_pe.to_string(index=False))
    print("="*70 + "\n")
    
    print("\n" + "="*90)
    print("SIDE-BY-SIDE VERGLEICH (Standard vs. Prompt Engineering)")
    print("="*90)
    print(side_by_side_df.to_string(index=False))
    print("="*90 + "\n")
    
    print("\n" + "="*90)
    print("ITERATION ANALYSIS - STANDARD APPROACH")
    print("="*90)
    if not iteration_df_std.empty:
        print(iteration_df_std.to_string(index=False))
    else:
        print("Keine Daten verfügbar")
    print("="*90 + "\n")
    
    print("\n" + "="*90)
    print("ITERATION ANALYSIS - PROMPT ENGINEERING APPROACH")
    print("="*90)
    if not iteration_df_pe.empty:
        print(iteration_df_pe.to_string(index=False))
    else:
        print("Keine Daten verfügbar")
    print("="*90 + "\n")

def main():
    """Hauptfunktion zur Auswertung der Judgements"""
    print("Lade Evaluationen aus beiden Ansätzen...")
    create_output_directory()
    
    # Lade Evaluationen aus beiden Verzeichnissen
    evaluations_standard = load_all_evaluations(EVALUATION_DIRS["standard"])
    evaluations_prompt_eng = load_all_evaluations(EVALUATION_DIRS["prompt_engineering"])
    
    print(f"\nGefundene Evaluationen (Standard):")
    for interp_model in INTERPRETATION_MODELS:
        for judge_model in JUDGE_MODELS:
            count = len(evaluations_standard[interp_model][judge_model])
            print(f"  {interp_model} bewertet von {judge_model}: {count}")
    
    print(f"\nGefundene Evaluationen (Prompt Engineering):")
    for interp_model in INTERPRETATION_MODELS:
        for judge_model in JUDGE_MODELS:
            count = len(evaluations_prompt_eng[interp_model][judge_model])
            print(f"  {interp_model} bewertet von {judge_model}: {count}")
    
    print("\nBerechne Statistiken für beide Ansätze...")
    # Aggregiere Evaluationen für beide Ansätze
    results_standard = aggregate_evaluations(evaluations_standard)
    results_prompt_eng = aggregate_evaluations(evaluations_prompt_eng)
    
    # Erstelle DataFrames für Standard Approach
    summary_df_std = create_summary_dataframe(results_standard)
    detailed_df_std = create_detailed_dataframe(results_standard)
    comparison_df_std = create_comparison_table(results_standard)
    agreement_df_std = analyze_judge_agreement(results_standard)
    iteration_df_std = create_iteration_analysis_dataframe(results_standard)
    
    # Erstelle DataFrames für Prompt Engineering Approach
    summary_df_pe = create_summary_dataframe(results_prompt_eng)
    detailed_df_pe = create_detailed_dataframe(results_prompt_eng)
    comparison_df_pe = create_comparison_table(results_prompt_eng)
    agreement_df_pe = analyze_judge_agreement(results_prompt_eng)
    iteration_df_pe = create_iteration_analysis_dataframe(results_prompt_eng)
    
    # Erstelle Vergleich zwischen den beiden Ansätzen
    approach_comp_df = create_approach_comparison(results_standard, results_prompt_eng)
    side_by_side_df = create_side_by_side_comparison(results_standard, results_prompt_eng)
    
    # Speichere Ergebnisse
    print("Speichere Ergebnisse...")
    save_results(results_standard, results_prompt_eng, 
                summary_df_std, summary_df_pe,
                detailed_df_std, detailed_df_pe,
                comparison_df_std, comparison_df_pe,
                agreement_df_std, agreement_df_pe,
                approach_comp_df, side_by_side_df,
                iteration_df_std, iteration_df_pe)
    
    # Zeige Zusammenfassung
    print_summary(comparison_df_std, comparison_df_pe, side_by_side_df,
                 iteration_df_std, iteration_df_pe)
    
    print(f"\nAlle Ergebnisse wurden in '{OUTPUT_DIR}' gespeichert!")
    print("\nNEU - Iterations-Analyse:")
    print("  - iteration_analysis_standard.csv")
    print("  - iteration_analysis_prompt_engineering.csv")
    print("\nStandard Approach:")
    print("  - full_results_standard.json")
    print("  - summary_standard.csv")
    print("  - detailed_scores_standard.csv")
    print("  - model_comparison_standard.csv")
    print("  - judge_agreement_standard.csv")
    print("\nPrompt Engineering Approach:")
    print("  - full_results_prompt_engineering.json")
    print("  - summary_prompt_engineering.csv")
    print("  - detailed_scores_prompt_engineering.csv")
    print("  - model_comparison_prompt_engineering.csv")
    print("  - judge_agreement_prompt_engineering.csv")
    print("\nVergleich:")
    print("  - approach_comparison.csv")
    print("  - side_by_side_comparison.csv")
    print("  - evaluation_results_complete.xlsx (alle Tabellen)")

if __name__ == "__main__":
    main()
