# results_display.py

from classical_ML_training import train_models

def display_results(datafile):
    results, _, _ = train_models(datafile)
    
    best_model_name = max(results, key=lambda k: results[k]['best_score'])
    best_model = results[best_model_name]

    print(f"Best Model: {best_model_name}")
    print(f"Best Model Accuracy: {best_model['best_score']:.4f}")
    print(f"Best Model Precision: {best_model['best_precision']:.4f}")
    print(f"Best Model Recall: {best_model['best_recall']:.4f}")
    print(f"Best Model F1-Score: {best_model['best_f1']:.4f}")
    print("\nClassification Report for Best Model:")
    print(best_model['best_classification'])
    
    # Detailed results for all models
    for model_name, model_results in results.items():
        print(f"\n------ {model_name} ------")
        print(f"Accuracy: {model_results['best_score']:.4f}")
        print(f"Precision: {model_results['best_precision']:.4f}")
        print(f"Recall: {model_results['best_recall']:.4f}")
        print(f"F1-Score: {model_results['best_f1']:.4f}")
        print("Best Parameters:", model_results['best_param'])

if __name__ == "__main__":
    datafile = "c2h4_final_data_phi_1.csv"
    display_results(datafile)

