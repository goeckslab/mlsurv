"""References section for report generation.

Dynamic bibliography with conditional citations based on trained models
and analysis methods, acknowledgments, and citation instructions.

Every citation carries a DOI where one exists. Two works have no DOI and
fall back to a stable URL: Mantel (1966), in the defunct *Cancer Chemotherapy
Reports*, links to PubMed; scikit-survival (JMLR, which mints no DOIs) links to
its JMLR page. The DOIs match the manuscript's model table (Table 2) and metric
table (Table 3) and are cross-checked against ``docs/evaluation.md``.
"""

from typing import Any

import pandas as pd

from mlsurv.reports.sections.base import BaseSection


# ---------------------------------------------------------------------------
# Citation database: key -> (APA citation, BibTeX)
# ---------------------------------------------------------------------------

_CITATION_DB: dict[str, tuple[str, str]] = {
    # --- Model citations ---
    'cox_1972': (
        'Cox, D. R. (1972). Regression models and life-tables. '
        '*Journal of the Royal Statistical Society: Series B*, 34(2), 187-202. '
        '<https://doi.org/10.1111/j.2517-6161.1972.tb00899.x>',
        '@article{cox1972regression,\n'
        '  title={Regression Models and Life-Tables},\n'
        '  author={Cox, David R},\n'
        '  journal={Journal of the Royal Statistical Society: Series B},\n'
        '  volume={34},\n'
        '  number={2},\n'
        '  pages={187--202},\n'
        '  year={1972},\n'
        '  doi={10.1111/j.2517-6161.1972.tb00899.x}\n'
        '}',
    ),
    'simon_2011': (
        'Simon, N., Friedman, J., Hastie, T., & Tibshirani, R. (2011). '
        'Regularization paths for Cox\'s proportional hazards model via '
        'coordinate descent. *Journal of Statistical Software*, 39(5), 1-13. '
        '<https://doi.org/10.18637/jss.v039.i05>',
        '@article{simon2011regularization,\n'
        '  title={Regularization Paths for Cox\'s Proportional Hazards Model '
        'via Coordinate Descent},\n'
        '  author={Simon, Noah and Friedman, Jerome and Hastie, Trevor '
        'and Tibshirani, Rob},\n'
        '  journal={Journal of Statistical Software},\n'
        '  volume={39},\n'
        '  number={5},\n'
        '  pages={1--13},\n'
        '  year={2011},\n'
        '  doi={10.18637/jss.v039.i05}\n'
        '}',
    ),
    'ishwaran_2008': (
        'Ishwaran, H., Kogalur, U. B., Blackstone, E. H., & Lauer, M. S. '
        '(2008). Random survival forests. *Annals of Applied Statistics*, '
        '2(3), 841-860. <https://doi.org/10.1214/08-AOAS169>',
        '@article{ishwaran2008random,\n'
        '  title={Random Survival Forests},\n'
        '  author={Ishwaran, Hemant and Kogalur, Udaya B and Blackstone, '
        'Eugene H and Lauer, Michael S},\n'
        '  journal={Annals of Applied Statistics},\n'
        '  volume={2},\n'
        '  number={3},\n'
        '  pages={841--860},\n'
        '  year={2008},\n'
        '  doi={10.1214/08-AOAS169}\n'
        '}',
    ),
    'hothorn_2006': (
        'Hothorn, T., Buhlmann, P., Dudoit, S., Molinaro, A., & van der Laan, '
        'M. J. (2006). Survival ensembles. *Biostatistics*, 7(3), 355-373. '
        '<https://doi.org/10.1093/biostatistics/kxj011>',
        '@article{hothorn2006survival,\n'
        '  title={Survival Ensembles},\n'
        '  author={Hothorn, Torsten and B{\\"u}hlmann, Peter and Dudoit, '
        'Sandrine and Molinaro, Annette and van der Laan, Mark J},\n'
        '  journal={Biostatistics},\n'
        '  volume={7},\n'
        '  number={3},\n'
        '  pages={355--373},\n'
        '  year={2006},\n'
        '  doi={10.1093/biostatistics/kxj011}\n'
        '}',
    ),
    'chen_2016': (
        'Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting '
        'system. In *Proceedings of the 22nd ACM SIGKDD International '
        'Conference on Knowledge Discovery and Data Mining* (pp. 785-794). '
        '<https://doi.org/10.1145/2939672.2939785>',
        '@inproceedings{chen2016xgboost,\n'
        '  title={XGBoost: A Scalable Tree Boosting System},\n'
        '  author={Chen, Tianqi and Guestrin, Carlos},\n'
        '  booktitle={Proceedings of the 22nd ACM SIGKDD International '
        'Conference on Knowledge Discovery and Data Mining},\n'
        '  pages={785--794},\n'
        '  year={2016},\n'
        '  doi={10.1145/2939672.2939785}\n'
        '}',
    ),
    'barnwal_2022': (
        'Barnwal, A., Cho, H., & Hocking, T. (2022). Survival regression with '
        'accelerated failure time model in XGBoost. *Journal of Computational '
        'and Graphical Statistics*, 31(4), 1292-1302. '
        '<https://doi.org/10.1080/10618600.2022.2067548>',
        '@article{barnwal2022survival,\n'
        '  title={Survival Regression with Accelerated Failure Time Model in '
        'XGBoost},\n'
        '  author={Barnwal, Avinash and Cho, Hyunsu and Hocking, Toby},\n'
        '  journal={Journal of Computational and Graphical Statistics},\n'
        '  volume={31},\n'
        '  number={4},\n'
        '  pages={1292--1302},\n'
        '  year={2022},\n'
        '  doi={10.1080/10618600.2022.2067548}\n'
        '}',
    ),
    'katzman_2018': (
        'Katzman, J. L., Shaham, U., Cloninger, A., Bates, J., Jiang, T., '
        '& Kluger, Y. (2018). DeepSurv: Personalized treatment recommender '
        'system using a Cox proportional hazards deep neural network. '
        '*BMC Medical Research Methodology*, 18(1), 24. '
        '<https://doi.org/10.1186/s12874-018-0482-1>',
        '@article{katzman2018deepsurv,\n'
        '  title={DeepSurv: Personalized Treatment Recommender System Using '
        'a Cox Proportional Hazards Deep Neural Network},\n'
        '  author={Katzman, Jared L and Shaham, Uri and Cloninger, Alexander '
        'and Bates, Jonathan and Jiang, Tingting and Kluger, Yuval},\n'
        '  journal={BMC Medical Research Methodology},\n'
        '  volume={18},\n'
        '  number={1},\n'
        '  pages={24},\n'
        '  year={2018},\n'
        '  doi={10.1186/s12874-018-0482-1}\n'
        '}',
    ),
    'lee_2018': (
        'Lee, C., Zame, W. R., Yoon, J., & van der Schaar, M. (2018). '
        'DeepHit: A deep learning approach to survival analysis with '
        'competing risks. In *Proceedings of the AAAI Conference on '
        'Artificial Intelligence*, 32(1). '
        '<https://doi.org/10.1609/aaai.v32i1.11842>',
        '@inproceedings{lee2018deephit,\n'
        '  title={DeepHit: A Deep Learning Approach to Survival Analysis '
        'with Competing Risks},\n'
        '  author={Lee, Changhee and Zame, William R and Yoon, Jinsung '
        'and van der Schaar, Mihaela},\n'
        '  booktitle={Proceedings of the AAAI Conference on Artificial '
        'Intelligence},\n'
        '  volume={32},\n'
        '  number={1},\n'
        '  year={2018},\n'
        '  doi={10.1609/aaai.v32i1.11842}\n'
        '}',
    ),
    'polsterl_2015': (
        'Polsterl, S., Navab, N., & Katouzian, A. (2015). Fast training of '
        'support vector machines for survival analysis. In *Machine Learning '
        'and Knowledge Discovery in Databases (ECML PKDD)*, Lecture Notes in '
        'Computer Science (Vol. 9285, pp. 243-259). '
        '<https://doi.org/10.1007/978-3-319-23525-7_15>',
        '@inproceedings{polsterl2015fast,\n'
        '  title={Fast Training of Support Vector Machines for Survival '
        'Analysis},\n'
        '  author={P{\\"o}lsterl, Sebastian and Navab, Nassir and Katouzian, '
        'Amin},\n'
        '  booktitle={Machine Learning and Knowledge Discovery in Databases '
        '(ECML PKDD)},\n'
        '  series={Lecture Notes in Computer Science},\n'
        '  volume={9285},\n'
        '  pages={243--259},\n'
        '  year={2015},\n'
        '  doi={10.1007/978-3-319-23525-7_15}\n'
        '}',
    ),
    'kalbfleisch_2002': (
        'Kalbfleisch, J. D., & Prentice, R. L. (2002). *The Statistical '
        'Analysis of Failure Time Data* (2nd ed.). Wiley. '
        '<https://doi.org/10.1002/9781118032985>',
        '@book{kalbfleisch2002statistical,\n'
        '  title={The Statistical Analysis of Failure Time Data},\n'
        '  author={Kalbfleisch, John D and Prentice, Ross L},\n'
        '  edition={2},\n'
        '  year={2002},\n'
        '  publisher={Wiley},\n'
        '  doi={10.1002/9781118032985}\n'
        '}',
    ),
    # --- Method citations ---
    'harrell_1982': (
        'Harrell, F. E., Califf, R. M., Pryor, D. B., Lee, K. L., & Rosati, '
        'R. A. (1982). Evaluating the yield of medical tests. *JAMA*, '
        '247(18), 2543-2546. <https://doi.org/10.1001/jama.1982.03320430047030>',
        '@article{harrell1982evaluating,\n'
        '  title={Evaluating the Yield of Medical Tests},\n'
        '  author={Harrell, Frank E and Califf, Robert M and Pryor, David B '
        'and Lee, Kerry L and Rosati, Robert A},\n'
        '  journal={JAMA},\n'
        '  volume={247},\n'
        '  number={18},\n'
        '  pages={2543--2546},\n'
        '  year={1982},\n'
        '  doi={10.1001/jama.1982.03320430047030}\n'
        '}',
    ),
    'uno_2011': (
        'Uno, H., Cai, T., Pencina, M. J., D\'Agostino, R. B., & Wei, L. J. '
        '(2011). On the C-statistics for evaluating overall adequacy of risk '
        'prediction procedures with censored survival data. *Statistics in '
        'Medicine*, 30(10), 1105-1117. <https://doi.org/10.1002/sim.4154>',
        '@article{uno2011cstatistics,\n'
        '  title={On the C-statistics for Evaluating Overall Adequacy of Risk '
        'Prediction Procedures with Censored Survival Data},\n'
        '  author={Uno, Hajime and Cai, Tianxi and Pencina, Michael J and '
        'D\'Agostino, Ralph B and Wei, Lee-Jen},\n'
        '  journal={Statistics in Medicine},\n'
        '  volume={30},\n'
        '  number={10},\n'
        '  pages={1105--1117},\n'
        '  year={2011},\n'
        '  doi={10.1002/sim.4154}\n'
        '}',
    ),
    'antolini_2005': (
        'Antolini, L., Boracchi, P., & Biganzoli, E. (2005). A time-dependent '
        'discrimination index for survival data. *Statistics in Medicine*, '
        '24(24), 3927-3944. <https://doi.org/10.1002/sim.2427>',
        '@article{antolini2005time,\n'
        '  title={A Time-dependent Discrimination Index for Survival Data},\n'
        '  author={Antolini, Laura and Boracchi, Patrizia and Biganzoli, '
        'Elia},\n'
        '  journal={Statistics in Medicine},\n'
        '  volume={24},\n'
        '  number={24},\n'
        '  pages={3927--3944},\n'
        '  year={2005},\n'
        '  doi={10.1002/sim.2427}\n'
        '}',
    ),
    'uno_2007': (
        'Uno, H., Cai, T., Tian, L., & Wei, L. J. (2007). Evaluating '
        'prediction rules for t-year survivors with censored regression '
        'models. *Journal of the American Statistical Association*, 102(478), '
        '527-537. <https://doi.org/10.1198/016214507000000149>',
        '@article{uno2007evaluating,\n'
        '  title={Evaluating Prediction Rules for t-Year Survivors with '
        'Censored Regression Models},\n'
        '  author={Uno, Hajime and Cai, Tianxi and Tian, Lu and Wei, '
        'Lee-Jen},\n'
        '  journal={Journal of the American Statistical Association},\n'
        '  volume={102},\n'
        '  number={478},\n'
        '  pages={527--537},\n'
        '  year={2007},\n'
        '  doi={10.1198/016214507000000149}\n'
        '}',
    ),
    'graf_1999': (
        'Graf, E., Schmoor, C., Sauerbrei, W., & Schumacher, M. (1999). '
        'Assessment and comparison of prognostic classification schemes for '
        'survival data. *Statistics in Medicine*, 18(17-18), 2529-2545. '
        '<https://doi.org/10.1002/(SICI)1097-0258(19990915/30)18:17/18%3C2529::AID-SIM274%3E3.0.CO;2-5>',
        '@article{graf1999assessment,\n'
        '  title={Assessment and Comparison of Prognostic Classification '
        'Schemes for Survival Data},\n'
        '  author={Graf, Erika and Schmoor, Claudia and Sauerbrei, Willi '
        'and Schumacher, Martin},\n'
        '  journal={Statistics in Medicine},\n'
        '  volume={18},\n'
        '  number={17-18},\n'
        '  pages={2529--2545},\n'
        '  year={1999},\n'
        '  doi={10.1002/(SICI)1097-0258(19990915/30)18:17/18<2529::AID-SIM274>3.0.CO;2-5}\n'
        '}',
    ),
    'steyerberg_2010': (
        'Steyerberg, E. W., Vickers, A. J., Cook, N. R., Gerds, T., Gonen, '
        'M., Obuchowski, N., Pencina, M. J., & Kattan, M. W. (2010). '
        'Assessing the performance of prediction models: A framework for '
        'traditional and novel measures. *Epidemiology*, 21(1), 128-138. '
        '<https://doi.org/10.1097/EDE.0b013e3181c30fb2>',
        '@article{steyerberg2010assessing,\n'
        '  title={Assessing the Performance of Prediction Models: A Framework '
        'for Traditional and Novel Measures},\n'
        '  author={Steyerberg, Ewout W and Vickers, Andrew J and Cook, Nancy '
        'R and Gerds, Thomas and Gonen, Mithat and Obuchowski, Nancy and '
        'Pencina, Michael J and Kattan, Michael W},\n'
        '  journal={Epidemiology},\n'
        '  volume={21},\n'
        '  number={1},\n'
        '  pages={128--138},\n'
        '  year={2010},\n'
        '  doi={10.1097/EDE.0b013e3181c30fb2}\n'
        '}',
    ),
    'demler_2015': (
        'Demler, O. V., Paynter, N. P., & Cook, N. R. (2015). Tests of '
        'calibration and goodness-of-fit in the survival setting. *Statistics '
        'in Medicine*, 34(10), 1659-1680. <https://doi.org/10.1002/sim.6428>',
        '@article{demler2015tests,\n'
        '  title={Tests of Calibration and Goodness-of-fit in the Survival '
        'Setting},\n'
        '  author={Demler, Olga V and Paynter, Nina P and Cook, Nancy R},\n'
        '  journal={Statistics in Medicine},\n'
        '  volume={34},\n'
        '  number={10},\n'
        '  pages={1659--1680},\n'
        '  year={2015},\n'
        '  doi={10.1002/sim.6428}\n'
        '}',
    ),
    'efron_1993': (
        'Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the '
        'Bootstrap*. Chapman and Hall/CRC. '
        '<https://doi.org/10.1201/9780429246593>',
        '@book{efron1993introduction,\n'
        '  title={An Introduction to the Bootstrap},\n'
        '  author={Efron, Bradley and Tibshirani, Robert J},\n'
        '  year={1993},\n'
        '  publisher={Chapman and Hall/CRC},\n'
        '  doi={10.1201/9780429246593}\n'
        '}',
    ),
    'lundberg_2017': (
        'Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to '
        'interpreting model predictions. In *Advances in Neural Information '
        'Processing Systems*, 30, 4766-4777. '
        '<https://doi.org/10.48550/arXiv.1705.07874>',
        '@inproceedings{lundberg2017unified,\n'
        '  title={A Unified Approach to Interpreting Model Predictions},\n'
        '  author={Lundberg, Scott M and Lee, Su-In},\n'
        '  booktitle={Advances in Neural Information Processing Systems},\n'
        '  volume={30},\n'
        '  pages={4766--4777},\n'
        '  year={2017},\n'
        '  doi={10.48550/arXiv.1705.07874}\n'
        '}',
    ),
    'breiman_2001': (
        'Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), '
        '5-32. <https://doi.org/10.1023/A:1010933404324>',
        '@article{breiman2001random,\n'
        '  title={Random Forests},\n'
        '  author={Breiman, Leo},\n'
        '  journal={Machine Learning},\n'
        '  volume={45},\n'
        '  number={1},\n'
        '  pages={5--32},\n'
        '  year={2001},\n'
        '  doi={10.1023/A:1010933404324}\n'
        '}',
    ),
    'altmann_2010': (
        'Altmann, A., Tolosi, L., Sander, O., & Lengauer, T. (2010). '
        'Permutation importance: A corrected feature importance measure. '
        '*Bioinformatics*, 26(10), 1340-1347. '
        '<https://doi.org/10.1093/bioinformatics/btq134>',
        '@article{altmann2010permutation,\n'
        '  title={Permutation Importance: A Corrected Feature Importance '
        'Measure},\n'
        '  author={Altmann, Andre and Tolosi, Laura and Sander, Oliver '
        'and Lengauer, Thomas},\n'
        '  journal={Bioinformatics},\n'
        '  volume={26},\n'
        '  number={10},\n'
        '  pages={1340--1347},\n'
        '  year={2010},\n'
        '  doi={10.1093/bioinformatics/btq134}\n'
        '}',
    ),
    'kaplan_1958': (
        'Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from '
        'incomplete observations. *Journal of the American Statistical '
        'Association*, 53(282), 457-481. '
        '<https://doi.org/10.1080/01621459.1958.10501452>',
        '@article{kaplan1958nonparametric,\n'
        '  title={Nonparametric Estimation from Incomplete Observations},\n'
        '  author={Kaplan, Edward L and Meier, Paul},\n'
        '  journal={Journal of the American Statistical Association},\n'
        '  volume={53},\n'
        '  number={282},\n'
        '  pages={457--481},\n'
        '  year={1958},\n'
        '  doi={10.1080/01621459.1958.10501452}\n'
        '}',
    ),
    'mantel_1966': (
        'Mantel, N. (1966). Evaluation of survival data and two new rank '
        'order statistics arising in its consideration. '
        '*Cancer Chemotherapy Reports*, 50(3), 163-170. '
        '<https://pubmed.ncbi.nlm.nih.gov/5910392/>',
        '@article{mantel1966evaluation,\n'
        '  title={Evaluation of Survival Data and Two New Rank Order '
        'Statistics Arising in Its Consideration},\n'
        '  author={Mantel, Nathan},\n'
        '  journal={Cancer Chemotherapy Reports},\n'
        '  volume={50},\n'
        '  number={3},\n'
        '  pages={163--170},\n'
        '  year={1966},\n'
        '  url={https://pubmed.ncbi.nlm.nih.gov/5910392/}\n'
        '}',
    ),
    # --- Guideline + software citations (always included) ---
    'collins_2024': (
        'Collins, G. S., et al. (2024). TRIPOD+AI statement: Updated '
        'guidance for reporting clinical prediction models that use '
        'regression or machine learning methods. *BMJ*, 385, e078378. '
        '<https://doi.org/10.1136/bmj-2023-078378>',
        '@article{collins2024tripod,\n'
        '  title={TRIPOD+AI Statement: Updated Guidance for Reporting '
        'Clinical Prediction Models That Use Regression or Machine Learning '
        'Methods},\n'
        '  author={Collins, Gary S and others},\n'
        '  journal={BMJ},\n'
        '  volume={385},\n'
        '  pages={e078378},\n'
        '  year={2024},\n'
        '  doi={10.1136/bmj-2023-078378}\n'
        '}',
    ),
    'polsterl_2020': (
        'Polsterl, S. (2020). scikit-survival: A library for time-to-event '
        'analysis built on top of scikit-learn. *Journal of Machine Learning '
        'Research*, 21(212), 1-6. <https://jmlr.org/papers/v21/20-729.html>',
        '@article{polsterl2020scikit,\n'
        '  title={scikit-survival: A Library for Time-to-Event Analysis '
        'Built on Top of scikit-learn},\n'
        '  author={P{\\"o}lsterl, Sebastian},\n'
        '  journal={Journal of Machine Learning Research},\n'
        '  volume={21},\n'
        '  number={212},\n'
        '  pages={1--6},\n'
        '  year={2020},\n'
        '  url={https://jmlr.org/papers/v21/20-729.html}\n'
        '}',
    ),
    'akiba_2019': (
        'Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). '
        'Optuna: A next-generation hyperparameter optimization framework. In '
        '*Proceedings of the 25th ACM SIGKDD International Conference on '
        'Knowledge Discovery & Data Mining* (pp. 2623-2631). '
        '<https://doi.org/10.1145/3292500.3330701>',
        '@inproceedings{akiba2019optuna,\n'
        '  title={Optuna: A Next-generation Hyperparameter Optimization '
        'Framework},\n'
        '  author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko '
        'and Ohta, Takeru and Koyama, Masanori},\n'
        '  booktitle={Proceedings of the 25th ACM SIGKDD International '
        'Conference on Knowledge Discovery \\& Data Mining},\n'
        '  pages={2623--2631},\n'
        '  year={2019},\n'
        '  doi={10.1145/3292500.3330701}\n'
        '}',
    ),
    'davidson_pilon_2019': (
        'Davidson-Pilon, C. (2019). lifelines: Survival analysis in Python. '
        '*Journal of Open Source Software*, 4(40), 1317. '
        '<https://doi.org/10.21105/joss.01317>',
        '@article{davidson_pilon2019lifelines,\n'
        '  title={lifelines: Survival Analysis in Python},\n'
        '  author={Davidson-Pilon, Cameron},\n'
        '  journal={Journal of Open Source Software},\n'
        '  volume={4},\n'
        '  number={40},\n'
        '  pages={1317},\n'
        '  year={2019},\n'
        '  doi={10.21105/joss.01317}\n'
        '}',
    ),
}

# Model key -> citation key mapping
# When adding a new citation: (1) add to _CITATION_DB, (2) add to _CITATION_TYPES,
# (3) if model citation, add to _MODEL_CITATIONS, (4) if always-included, add to
# _ALWAYS_INCLUDED, (5) if conditional method citation, add detection logic to
# _detect_used_citations.
_MODEL_CITATIONS: dict[str, str] = {
    'coxph': 'cox_1972',
    'coxnet': 'simon_2011',
    'rsf': 'ishwaran_2008',
    'gbsa': 'hothorn_2006',
    'xgbcox': 'chen_2016',
    'xgbaft': 'barnwal_2022',
    'deepsurv': 'katzman_2018',
    'deephit': 'lee_2018',
    'fssvm': 'polsterl_2015',
    'weibull_aft': 'kalbfleisch_2002',
}

# Citation type classification
_CITATION_TYPES: dict[str, str] = {
    'cox_1972': 'Model',
    'simon_2011': 'Model',
    'ishwaran_2008': 'Model',
    'hothorn_2006': 'Model',
    'chen_2016': 'Model',
    'barnwal_2022': 'Model',
    'katzman_2018': 'Model',
    'lee_2018': 'Model',
    'polsterl_2015': 'Model',
    'kalbfleisch_2002': 'Model',
    'harrell_1982': 'Method',
    'uno_2011': 'Method',
    'antolini_2005': 'Method',
    'uno_2007': 'Method',
    'graf_1999': 'Method',
    'steyerberg_2010': 'Method',
    'demler_2015': 'Method',
    'efron_1993': 'Method',
    'lundberg_2017': 'Method',
    'breiman_2001': 'Method',
    'altmann_2010': 'Method',
    'kaplan_1958': 'Method',
    'mantel_1966': 'Method',
    'collins_2024': 'Guideline',
    'polsterl_2020': 'Software',
    'akiba_2019': 'Software',
    'davidson_pilon_2019': 'Software',
}

# Always-included citation keys
_ALWAYS_INCLUDED = frozenset({
    'collins_2024',
    'polsterl_2020',
    'akiba_2019',
    'davidson_pilon_2019',
})


def _eval_has_metric(models, column: str) -> bool:
    """True if any model's evaluation concordance carries a value in ``column``.

    Used to cite a metric's source only when that metric was actually computed
    for at least one model (e.g. calibration columns are absent for models
    without ``predict_survival_function``).
    """
    for result in models.values():
        evaluation = getattr(result, 'evaluation', None)
        if evaluation is None:
            continue
        concordance = getattr(evaluation, 'concordance', None)
        if concordance is None:
            continue
        try:
            if column in concordance.columns and concordance[column].notna().any():
                return True
        except (AttributeError, TypeError):
            # Mock/placeholder evaluation in tests — treat as "metric absent".
            continue
    return False


def _eval_has_auc(models) -> bool:
    """True if any model's evaluation has a non-empty time-dependent AUC table."""
    for result in models.values():
        evaluation = getattr(result, 'evaluation', None)
        if evaluation is None:
            continue
        auc_t = getattr(evaluation, 'auc_t', None)
        try:
            if auc_t is not None and not auc_t.empty:
                return True
        except (AttributeError, TypeError):
            continue
    return False


def _eval_has_perm_importance(models) -> bool:
    """True if any model carries permutation-importance values (``perm_mean``).

    Distinguishes the default permutation importance (Breiman 2001) from the
    PIMP significance test (Altmann 2010): the former is the ``perm_mean``
    column on ``evaluation.importance``; the latter is a separate test table.
    """
    for result in models.values():
        evaluation = getattr(result, 'evaluation', None)
        if evaluation is None:
            continue
        importance = getattr(evaluation, 'importance', None)
        if importance is None:
            continue
        try:
            if 'perm_mean' in importance.columns and importance['perm_mean'].notna().any():
                return True
        except (AttributeError, TypeError):
            continue
    return False


class ReferencesSection(BaseSection):
    """
    Combined references section with dynamic bibliography.

    Citations are included conditionally based on which models were
    trained and which analysis methods were used. Always-included
    citations cover foundational software and guidelines.
    """

    name: str = "references"
    title: str = "References"

    def is_available(self) -> bool:
        """Always available."""
        return True

    def generate(self) -> dict[str, Any]:
        """Generate combined references section content."""
        used_keys = self._detect_used_citations()

        # Store citations data
        rows = []
        for key in used_keys:
            apa, _ = _CITATION_DB[key]
            rows.append({
                'key': key,
                'citation': apa,
                'type': _CITATION_TYPES.get(key, 'Other'),
            })
        self.data['citations'] = pd.DataFrame(rows)

        parts: list[str] = []

        # --- Acknowledgments ---
        parts.append(self._generate_acknowledgments())
        parts.append("")
        parts.append("---")
        parts.append("")

        # --- Citing mlsurv ---
        parts.append(self._generate_mlsurv_citation())
        parts.append("")
        parts.append("---")
        parts.append("")

        # --- Dynamic bibliography ---
        parts.append(self._generate_bibliography(used_keys))

        return {
            'markdown': '\n'.join(parts),
            'figures': [],
        }

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect_used_citations(self) -> list[str]:
        """Detect which citations are relevant to this analysis."""
        used = set(_ALWAYS_INCLUDED)

        try:
            models = self.learner.models
            pimp_results = self.learner.pimp_importance_test_results
        except AttributeError as exc:
            self._log_recoverable_exception(
                exc,
                "references: could not access learner state for citation detection",
                warn_user=True,
                user_message="Could not detect analysis methods for dynamic bibliography.",
            )
            return self._sort_citation_keys(used)

        # Model citations
        for model_key, cite_key in _MODEL_CITATIONS.items():
            if model_key in models:
                used.add(cite_key)

        # Method citations — gate each on the metric actually being present, so
        # the bibliography cites only the methods this analysis used. The
        # mapping mirrors the manuscript's evaluation-metric table.
        has_eval = any(m.evaluation is not None for m in models.values())
        if has_eval:
            # Harrell's C, KM risk curves, and the log-rank separation test are
            # produced for every evaluated model.
            used.add('harrell_1982')  # Harrell's C-index
            used.add('kaplan_1958')   # KM curves
            used.add('mantel_1966')   # Log-rank
            # The remaining discrimination/calibration metrics are optional;
            # cite each only when its concordance column carries a value.
            if _eval_has_metric(models, 'c_index_ipcw_test'):
                used.add('uno_2011')        # Uno's IPCW C-index
            if _eval_has_metric(models, 'c_index_td_test'):
                used.add('antolini_2005')   # Antolini's time-dependent C
            if _eval_has_metric(models, 'integrated_brier_test'):
                used.add('graf_1999')       # Brier score / IBS
            if (_eval_has_metric(models, 'cal_slope_test')
                    or _eval_has_metric(models, 'cal_intercept_test')):
                used.add('steyerberg_2010')  # Calibration slope/intercept
            if _eval_has_metric(models, 'gnd_test'):
                used.add('demler_2015')     # GND calibration test
            if _eval_has_auc(models):
                used.add('uno_2007')        # Time-dependent AUC(t)

        has_boot = any(
            m.bootstrap is not None
            or (m.evaluation is not None and m.evaluation.bootstrap is not None)
            for m in models.values()
        )
        if has_boot:
            used.add('efron_1993')

        # Default permutation importance (Breiman 2001) — the perm_mean bars.
        if _eval_has_perm_importance(models):
            used.add('breiman_2001')

        # SHAP: check if any model has shap_values
        for result in models.values():
            if getattr(result, 'shap_values', None) is not None:
                used.add('lundberg_2017')
                break

        # PIMP significance test (Altmann 2010) — distinct from the plain
        # permutation importance above. Dict truthiness: empty == never ran.
        if pimp_results:
            used.add('altmann_2010')

        return self._sort_citation_keys(used)

    @staticmethod
    def _sort_citation_keys(keys) -> list[str]:
        """Sort citation keys alphabetically by first author surname."""
        def _sort_key(key: str) -> str:
            apa = _CITATION_DB[key][0]
            return apa.split(',')[0].split('.')[-1].strip().lower()

        return sorted(keys, key=_sort_key)

    # ------------------------------------------------------------------
    # Content generators
    # ------------------------------------------------------------------

    def _generate_acknowledgments(self) -> str:
        """Generate the acknowledgments subsection."""
        lines = [
            "### Acknowledgments",
            "",
            "mlsurv is built on a foundation of established open-source "
            "libraries.",
            "",
            "**scikit-survival** provides the foundational survival analysis "
            "models and metrics that power mlsurv's core functionality. Its "
            "scikit-learn-compatible design influenced our model interface, "
            "enabling seamless integration with existing machine learning "
            "workflows.",
            "",
            "**lifelines** powers Kaplan-Meier estimation, log-rank tests, "
            "and Cox regression diagnostics, providing robust non-parametric "
            "and semi-parametric survival analysis capabilities.",
            "",
            "**scikit-learn** shapes mlsurv's interface through its Pipeline, "
            "BaseEstimator, and cross-validation abstractions, enabling "
            "consistent model composition and evaluation patterns.",
            "",
            "**Optuna** enables mlsurv's intelligent hyperparameter "
            "optimization through its efficient tree-structured Parzen "
            "estimator algorithm.",
            "",
            "PyCaret's low-code philosophy inspired mlsurv's user-friendly "
            "interface design.",
            "",
        ]

        # Runtime deps
        lines.append(
            "mlsurv also relies on **NumPy**, **pandas**, and **Plotly** "
            "as core runtime dependencies."
        )

        # Conditional XGBoost acknowledgment
        if self._has_xgboost_models():
            lines.append("")
            lines.append(
                "**XGBoost** powers the gradient-boosted survival models "
                "(XGBCox, XGBAFT) used in this analysis."
            )

        # Conditional DL acknowledgment
        if self._has_deep_learning_models():
            lines.append("")
            lines.append(
                "**PyTorch** and **pycox** power the deep learning survival "
                "models (DeepSurv, DeepHit) used in this analysis."
            )

        lines.append("")
        lines.append(
            "We thank the developers and contributors of these projects."
        )
        return '\n'.join(lines)

    @staticmethod
    def _generate_mlsurv_citation() -> str:
        """Generate the 'Citing mlsurv' block."""
        return (
            "### Citing mlsurv\n"
            "\n"
            "If you use mlsurv in your research, please cite:\n"
            "\n"
            "Goecks Lab at Moffitt Cancer Center. (2026). "
            "*mlsurv: Low-Code Survival Machine Learning in Python* "
            "[Computer software]. https://github.com/goeckslab/mlsurv\n"
            "\n"
            "<details>\n"
            "<summary>BibTeX</summary>\n"
            "\n"
            "```bibtex\n"
            "@software{mlsurv,\n"
            "  title = {mlsurv: Low-Code Survival Machine Learning in Python},\n"
            "  author = {Goecks Lab at Moffitt Cancer Center},\n"
            "  year = {2026},\n"
            "  url = {https://github.com/goeckslab/mlsurv}\n"
            "}\n"
            "```\n"
            "\n"
            "</details>"
        )

    @staticmethod
    def _generate_bibliography(used_keys: list[str]) -> str:
        """Generate the bibliography with categorized sections."""
        # Partition keys by citation type
        by_type: dict[str, list[str]] = {}
        for key in used_keys:
            cite_type = _CITATION_TYPES.get(key, 'Other')
            by_type.setdefault(cite_type, []).append(key)

        lines = ["### Bibliography", ""]

        for section_title, cite_type in [
            ("Model References", "Model"),
            ("Method References", "Method"),
            ("Reporting Guidelines", "Guideline"),
            ("Software References", "Software"),
        ]:
            keys = by_type.get(cite_type, [])
            if not keys:
                continue
            lines.append(f"#### {section_title}")
            lines.append("")
            for key in keys:
                apa, bibtex = _CITATION_DB[key]
                lines.append(apa)
                lines.append("")
                lines.append("<details>")
                lines.append("<summary>BibTeX</summary>")
                lines.append("")
                lines.append("```bibtex")
                lines.append(bibtex)
                lines.append("```")
                lines.append("")
                lines.append("</details>")
                lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("*Report generated with mlsurv*")
        return '\n'.join(lines)
