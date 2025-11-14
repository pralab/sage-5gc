# thales_project

In ml_models_new ci sono i nuovi 4 modelli riadattati e in final_datasets i 3 dataset.csv utilizzati. Bisogna prima eseguire evaluation_models_new.py per addestrare e salvare i modelli e mostrare le prestazioni generali. Poi si può eseguire evaluate_robusteness.py per calcolare e salvare il grafico sulla robustezza.

- NOTA: IsolationForest e MLP ottimizzano la threshold sul test set, penso che in teoria si dovrebbe utilizzare un validation set perchè così pompa un po' le prestazioni secondo me, infatti senza ottimizzazione fanno abbastanza schifo, infatti risultano i meno robusti. Io li ho utilizzati con questa threshold fissa una volta calcolata.
