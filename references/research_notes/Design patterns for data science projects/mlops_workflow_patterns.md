# MLOps Workflow and Process Patterns for Data Science Projects

## Key Question 1: What lifecycle/process frameworks and maturity models exist, and what phases do they define?

### Takeaway
Five named frameworks dominate: CRISP-DM (1996–1999, six phases, still the de facto standard despite stalled updates), Microsoft TDSP (agile five-stage lifecycle extending CRISP-DM), Google's three-level MLOps automation model (manual → ML pipeline automation → CI/CD pipeline automation, with Continuous Training as the pivotal capability), Microsoft's five-level MLOps maturity model (No MLOps → full automated operations), and ThoughtWorks' CD4ML (continuous delivery extended to the code/data/model triad).

### Cited Findings

**CRISP-DM (Cross-Industry Standard Process for Data Mining)**
- CRISP-DM defines six phases: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment; the process is cyclical rather than strictly linear, with frequent movement between phases — [Wikipedia: CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
- History: conceived in 1996, became an EU project under ESPRIT funding in 1997, led by five companies — Integral Solutions Ltd (ISL), Teradata, Daimler AG, NCR Corporation, and insurer OHRA; first methodology version presented March 1999 at the 4th CRISP-DM SIG Workshop in Brussels and published as a step-by-step guide in 1999 — [Wikipedia: CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
- De facto standard status: KDnuggets polls in 2002, 2004, 2007, and 2014 consistently ranked CRISP-DM as the leading methodology; a 2009 review called it the "de facto standard for developing data mining and knowledge discovery projects" — [Wikipedia: CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
- Criticism/limitations: CRISP-DM lacks project-management capabilities; the CRISP-DM 2.0 update effort (2006–2008) was ultimately unsuccessful and the original website and 2.0 SIG no longer exist. Successors include IBM's ASUM-DM (2015, adds agile principles) and bizML (2024, HBR-published, ML-specific) — [Wikipedia: CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)

**Microsoft Team Data Science Process (TDSP)**
- TDSP is a methodology that applies Agile principles to deliver data science solutions; its backbone is the "data science lifecycle" of five iterative stages: Business Understanding; Data Acquisition and Understanding; Modeling; Deployment; Customer (stakeholder) Acceptance — [datascience-pm.com: TDSP](https://www.datascience-pm.com/tdsp/); stage docs at [Microsoft Learn: lifecycle-acceptance](https://learn.microsoft.com/en-us/azure/architecture/data-science-process/lifecycle-acceptance) and [MicrosoftDocs architecture-center source](https://github.com/microsoftdocs/architecture-center/blob/main/docs/data-science-process/lifecycle-data.md)
- Mapping to CRISP-DM: TDSP's "Data Acquisition and Understanding" merges data understanding/cleaning; "Modeling" combines feature engineering with model training and evaluation; "Customer Acceptance" (customer validates the pipeline, model, and production deployment against business objectives) is a phase not explicitly covered by CRISP-DM — [datascience-pm.com: TDSP](https://www.datascience-pm.com/tdsp/)
- Note: the original TDSP overview pages on learn.microsoft.com have been retired (the English overview/lifecycle URLs now redirect to the Cloud Adoption Framework AI plan page), and the [Azure/Microsoft-TDSP GitHub repo README](https://raw.githubusercontent.com/Azure/Microsoft-TDSP/master/README.md) now carries only a deprecation notice pointing to aka.ms/tdsp — observed during this research (Oct 2026)

**Google MLOps maturity levels** (canonical article: "MLOps: Continuous delivery and automation pipelines in machine learning")
- Framing problem: "only a small fraction of a real-world ML system is composed of the ML code"; the surrounding infrastructure (configuration, automation, data collection/validation, testing, monitoring, serving) is vast — [Google Cloud Architecture Center](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- **Level 0 — Manual process**: every step (data extraction/analysis, preparation, training, evaluation, validation) executed manually, typically in interactive notebooks; script-driven; disconnect between data scientists and engineers creates training-serving skew; infrequent releases (a few per year); no CI, no CD, no performance monitoring — [Google Cloud](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- **Level 1 — ML pipeline automation**: the training pipeline itself is automated, achieving "continuous training of the model in production using fresh data based on live pipeline triggers"; requires new components: automated data validation and model validation, a feature store (centralized feature definitions shared between training and serving to prevent training-serving skew), ML metadata management (lineage of artifacts, parameters, metrics), and pipeline triggers (on schedule, on new data, on performance degradation / concept drift); code is modularized into reusable containerized components; the same pipeline runs in dev and production — [Google Cloud](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- **Level 2 — CI/CD pipeline automation**: automated build/test/package of pipeline components on commit; continuous delivery of whole pipeline implementations to target environments; testing additions include unit tests for feature-engineering logic, model convergence checks, integration tests of the pipeline, prediction-service API verification, and load testing — [Google Cloud](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- Continuous Training (CT) is the defining capability separating ML delivery from traditional software deployment: automated retraining triggered by new data, concept drift, monitored performance degradation, or schedule — [Google Cloud](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

**Microsoft MLOps maturity model** (Azure Architecture Center)
- Five levels of technical capability, each assessed across People / Model creation / Model release / Application integration: **Level 0 "No MLOps"** (manual builds, deployments, training; no centralized tracking), **Level 1 "DevOps but no MLOps"** (automated application builds and tests, version-controlled code, but experiments untracked and model handoff manual), **Level 2 "Automated training"** (automated, reproducible training; centralized experiment tracking; training code and models version controlled; releases still manual), **Level 3 "Automated model deployment"** (automatic releases via CI/CD; A/B testing integrated; full traceability from deployment back to original data), **Level 4 "Full MLOps automated operations"** (full automation and monitoring; drift/regression signals automatically trigger retraining; approaching zero downtime) — [Microsoft Learn: MLOps maturity model](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model)
- The model is meant to "assess your current state and plan incremental progress"; organizations often show characteristics of multiple levels simultaneously — progression is a continuum, not discrete stages — [Microsoft Learn](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model)

**CD4ML (Continuous Delivery for Machine Learning)**
- Definition: "a software engineering approach in which a cross-functional team produces machine learning applications based on code, data, and models in small and safe increments that can be reproduced and reliably released at any time, in short adaptation cycles"; authored by Danilo Sato, Arif Wider, and Christoph Windheuser (Thoughtworks), published September 19, 2019 on martinfowler.com — [martinfowler.com: CD4ML](https://martinfowler.com/articles/cd4ml.html)
- Core premise: ML applications change along **three axes — code, model, and data** — each requiring its own versioning and management strategy; technical components include discoverable/accessible data, reproducible model training (e.g., via DVC pipelines), model serving patterns (embedded model, model as separate service, model-as-data ingested at runtime), a testing pyramid spanning code/data/model, experiment tracking (Git branches + MLflow Tracking), deployment pipelines with governance gates (bias/fairness checks), and monitoring/observability closing the feedback loop — [martinfowler.com: CD4ML](https://martinfowler.com/articles/cd4ml.html)

**ml-ops.org (INNOQ) synthesis**
- Defines the CI/CD/CT vocabulary: CI extends testing to "data and models" alongside code; CD deploys ML model prediction services; **CT is "unique to ML systems"** (automatic retraining and redeployment); CM (Continuous Monitoring) covers production data and metrics; also restates the three automation maturity levels (manual process → ML pipeline automation → CI/CD pipeline automation), mirroring Google's model — [ml-ops.org: MLOps Principles](https://ml-ops.org/content/mlops-principles)

### Inferences
- The Google 3-level model and Microsoft 5-level model describe the same progression at different granularity: Microsoft's levels 2–4 decompose Google's levels 1–2 (automated training vs. automated deployment vs. automated retraining-on-drift).
- CRISP-DM covers the pre-deployment analytics process; the MLOps maturity models begin roughly where CRISP-DM's "Deployment" phase ends — teams commonly cite CRISP-DM/TDSP for project phases and Google/Microsoft maturity models for operational capability.

### Gaps
- Microsoft has retired the detailed TDSP documentation set from learn.microsoft.com (English pages redirect to the Cloud Adoption Framework), so the authoritative description of TDSP's four components (lifecycle definition, standardized project structure, infrastructure/resources, tools/utilities) could not be verified against a live first-party page; only secondary sources (datascience-pm.com) and the archived GitHub docs source remain accessible.

## Key Question 2: What are the standard workflow patterns (experiment tracking, model registry, data versioning, orchestration, reproducibility)?

### Takeaway
Five recurring workflow patterns are well documented: (1) experiment tracking as runs grouped into experiments with logged params/metrics/artifacts; (2) a model registry with versioned models promoted via mutable aliases/stages (champion/challenger); (3) data versioning via small Git-committed metafiles pointing at content-addressed data in remote storage (DVC); (4) pipeline-as-DAG orchestration with dependency-resolved execution; (5) reproducibility via versioned data + pinned environments + containerized pipeline components.

### Cited Findings

**Experiment tracking pattern**
- CD4ML prescribes managing parallel model experiments with Git branches plus a tracking tool (MLflow Tracking) that captures parameters, metrics, and performance data for each run — [martinfowler.com: CD4ML](https://martinfowler.com/articles/cd4ml.html)
- MLflow's registry maintains lineage from each registered model version back to "which MLflow experiment and run produced the model," making run/experiment tracking the substrate of the full lifecycle — [MLflow Model Registry docs](https://mlflow.org/docs/latest/ml/model-registry/)
- ml-ops.org lists experiment tracking and versioning of "ML training scripts, models, and datasets" as first-class MLOps versioning principles (covering data-prep pipelines, feature stores, datasets, hyperparameters, metadata) — [ml-ops.org: MLOps Principles](https://ml-ops.org/content/mlops-principles)
- Microsoft's maturity model uses "experiment results are tracked" + "training code and models are both version controlled" as the gate distinguishing Level 2 (Automated training) from Levels 0–1 — [Microsoft Learn: MLOps maturity model](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model)

**Model registry + promotion pattern**
- MLflow Model Registry concepts: a **Registered Model** has "a unique name, contains versions, aliases, tags, and other metadata"; each newly registered model under a name increments a **Model Version** (starting at version 1); **aliases** are "a mutable, named reference to a particular version of a registered model" (e.g., `@champion`), and promotion is done by reassigning the alias rather than by traditional stage transitions; **tags** (e.g., `validation_status:approved`) label status — [MLflow Model Registry docs](https://mlflow.org/docs/latest/ml/model-registry/)
- Microsoft Level 3 adds artifact "promotion across workspaces by using Machine Learning registries" and Level 4 makes "model promotion policy-based and automated" — i.e., registry-mediated promotion is the release mechanism at the top maturity levels — [Microsoft Learn: MLOps maturity model](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model)

**Data versioning pattern (DVC)**
- DVC stores "simple metafiles" (`.dvc` files) in Git as pointers to data held in a content-addressable local cache (hash-identified, preventing duplication) and remote storage (S3, GCS, HDFS, etc.); `git checkout` + `dvc checkout` switches data versions in sync with code; `dvc push`/`dvc pull` synchronize cache with remote storage; the result is "a single history for data, code, and ML models that you can traverse" — [DVC docs: Versioning Data and Models](https://doc.dvc.org/use-cases/versioning-data-and-models)
- DVC pipelines: `dvc.yaml` defines stages and dependencies; `dvc repro` re-executes the pipeline matching correct data versions to code versions via Git history — [DVC docs](https://doc.dvc.org/use-cases/versioning-data-and-models); CD4ML names DVC as its reference tool for reproducible model training — [martinfowler.com: CD4ML](https://martinfowler.com/articles/cd4ml.html)

**Pipeline orchestration (DAG) pattern**
- Kedro's model of the pattern: a **node** is "a wrapper for a pure Python function that names the inputs and outputs of that function"; a **pipeline** "determines the node execution order by resolving dependencies" (i.e., a DAG) rather than running nodes in listed order; the **Data Catalog** is a central registry of all project datasets with "built-in datasets for several file types and file systems, so you don't have to write the logic for reading/writing data" — [Kedro docs: Kedro concepts](https://docs.kedro.org/en/stable/getting-started/kedro_concepts/)
- Google Level 1 requires the training workflow be an orchestrated, automated pipeline of modular, containerized, reusable components (vs. a monolithic notebook/script) — [Google Cloud](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- CD4ML orchestrates combined ML + application deployment pipelines via a CD tool (GoCD in the article), with governance gates in the pipeline — [martinfowler.com: CD4ML](https://martinfowler.com/articles/cd4ml.html)

**Standardized project structure pattern**
- Kedro projects follow a standardized template separating `conf/` (catalog, parameters, logging config), `src/`, `data/`, `tests/`, and `notebooks/`, with configuration kept apart from code and data — [Kedro docs: Kedro concepts](https://docs.kedro.org/en/stable/getting-started/kedro_concepts/)

**Reproducibility pattern**
- ml-ops.org's reproducibility principle: identical results must be ensured across data collection, feature engineering, training, and deployment via data backups/versioning, documented transformations, and environment consistency — [ml-ops.org: MLOps Principles](https://ml-ops.org/content/mlops-principles)
- Google Level 1 relies on containerized pipeline components and ML metadata (artifact lineage, parameters, evaluation metrics) "for reproducibility and debugging"; Microsoft Level 2 is defined by a "fully managed and traceable" training environment where the "model is easy to reproduce," with environments managed via environment definitions — [Google Cloud](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning); [Microsoft Learn](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model)

### Inferences
- The common thread across tracking, registry, and data versioning is lineage: every production artifact should be traceable to the exact code commit, data version, parameters, and run that produced it (Microsoft Level 3's "full traceability from deployment back to original data" is the explicit target state).
- Airflow, Prefect, and Dagster implement the same DAG-orchestration pattern Kedro describes (dependency-resolved task graphs); Kedro was cited here because its docs state the pattern most explicitly in data-science terms.

### Gaps
- Explicit canonical sourcing for random-seed fixing as a named reproducibility practice was not found in the fetched framework documents (it appears in tool-level docs and practitioner guides); the frameworks instead emphasize data/version/environment determinism.
- MLflow's historical stage-based promotion (Staging → Production → Archived) and its formal deprecation in favor of aliases (circa MLflow 2.9) is consistent with the current registry docs ("rather than using traditional stage transitions, you update which model version an alias references"), but the exact deprecation release note was not fetched.

## Key Question 3: What testing patterns exist for data science (data validation, model validation gates, behavioral testing, ML Test Score)?

### Takeaway
Testing in ML splits into three scopes — data tests, model tests, infrastructure tests — codified by the ML Test Score rubric (Breck et al., 28 tests, 4 sections, min-across-sections scoring). Declarative data validation (Great Expectations' "unit tests for data"), automated model validation gates before registry promotion, and behavioral testing beyond held-out accuracy (CheckList, ACL 2020 Best Paper) are the named patterns.

### Cited Findings

**ML Test Score rubric**
- Canonical paper: Breck, Cai, Nielsen, Salib, Sculley (2017), "The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction," IEEE Big Data; presents "28 specific tests and monitoring needs, drawn from experience with a wide range of production ML systems" as "a road-map to improve production readiness and pay down ML technical debt" — [Google Research publication page](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/)
- Earlier version: "What's your ML test score? A rubric for ML production systems," same authors, Reliable Machine Learning in the Wild workshop, NIPS 2016 — [Google Research](https://research.google/pubs/whats-your-ml-test-score-a-rubric-for-ml-production-systems/)
- Scoring mechanics: 0.5 points per test performed manually, 1 point if automated and run repeatedly; the **minimum** score across the four sections (Data, Model, ML Infrastructure, Monitoring) determines the overall score (ranges 0 to >5) — forcing balanced coverage rather than excellence in one area — [ml-ops.org: MLOps Principles](https://ml-ops.org/content/mlops-principles)

**Data validation pattern**
- Great Expectations pattern: **Expectations** are "verifiable assertions about data" functioning like unit-test assertions in "a flexible, declarative language"; grouped into **Expectation Suites**; **Validation Definitions** bind batches of data to suites; **Checkpoints** run validations in production and trigger **Actions** (Slack/email alerts, pipeline hooks); **Data Docs** auto-generate human-readable documentation of suites and validation results — [Great Expectations docs: GX overview](https://docs.greatexpectations.io/docs/core/introduction/gx_overview)
- Google Level 1 requires automated **data validation** (schema + statistical anomaly detection) before retraining, as a mandatory pipeline step — [Google Cloud](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- CD4ML's testing pyramid validates data schemas as one of its distinct test classes — [martinfowler.com: CD4ML](https://martinfowler.com/articles/cd4ml.html)

**Model validation gates**
- Google Level 1 requires offline **model validation**: automatically comparing the newly trained model against the current baseline before deployment — [Google Cloud](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- CD4ML includes model quality threshold tests and fairness evaluation across data slices, plus governance gates for bias/fairness in the deployment pipeline — [martinfowler.com: CD4ML](https://martinfowler.com/articles/cd4ml.html)
- MLflow tags such as `validation_status:approved` mark validation state on model versions in the registry workflow — [MLflow Model Registry docs](https://mlflow.org/docs/latest/ml/model-registry/)

**Behavioral testing (CheckList)**
- Ribeiro, Wu, Guestrin, Singh (2020), "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList," ACL 2020 (Best Overall Paper): "measuring held-out accuracy... often overestimates the performance of NLP models"; CheckList is "a task-agnostic methodology for testing NLP models" using "a matrix of general linguistic capabilities and test types"; in a user study, practitioners with CheckList "created twice as many tests, and found almost three times as many bugs as users without it" — [ACL Anthology 2020.acl-main.442](https://aclanthology.org/2020.acl-main.442/)

**Test scopes / pipeline smoke tests**
- ml-ops.org organizes ML testing into three scopes — features/data tests, model development tests, ML infrastructure tests — including data validation, feature-importance testing, model staleness checks, fairness/bias assessment, training reproducibility verification, and pipeline integration testing — [ml-ops.org: MLOps Principles](https://ml-ops.org/content/mlops-principles)
- Google Level 2 enumerates the CI test additions for training pipelines: unit tests for feature-engineering logic, model convergence checks, integration tests of the full pipeline, prediction-service API verification, and load tests — [Google Cloud](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- CD4ML frames the overall approach as separate testing pyramids for code, data, and models, combined — [martinfowler.com: CD4ML](https://martinfowler.com/articles/cd4ml.html)

### Inferences
- The "smoke test on the training pipeline" practice (run the full pipeline on a tiny data sample in CI) is the practical implementation of Google Level 2's "integration testing of the pipeline" and the ML Test Score's infrastructure tests, though no fetched source names it "smoke test" verbatim.
- CheckList's MFT/invariance/directional-expectation taxonomy generalizes beyond NLP into the broader "behavioral model testing" pattern now echoed in ML testing guides.

### Gaps
- The ACL Anthology landing page confirms CheckList's citation and findings but does not enumerate the three test types (Minimum Functionality Test, Invariance, Directional Expectation) on the abstract page itself; that detail is in the paper PDF, which was not fetched.
- The full list of the 28 ML Test Score tests (7 per section) was not reproduced on the fetched pages; only the section structure and scoring were verifiable. The paper PDF is the canonical source for the item-level list.
- pandera (dataframe schema validation in code, the lightweight alternative to Great Expectations) was not independently verified against its docs in this research pass.

## Key Question 4: What CI/CD patterns apply to ML (continuous training, GitOps for models, champion/challenger)?

### Takeaway
ML extends CI/CD with CT (Continuous Training) and CM (Continuous Monitoring): CI tests data and models in addition to code; CD delivers whole training pipelines, not just binaries; retraining is trigger-driven (schedule, new data, drift); and release uses progressive patterns — shadow deployment, A/B testing, and champion/challenger promotion via registry aliases.

### Cited Findings
- Redefined CI/CD for ML: CI "extends testing to include data and models alongside code"; CD "automatically deploys ML model prediction services"; **CT** is "unique to ML systems" (automatic retraining for redeployment); **CM** monitors production data and performance metrics — [ml-ops.org: MLOps Principles](https://ml-ops.org/content/mlops-principles)
- In Google's Level 1+, what gets deployed is not a model binary but a **training pipeline** that runs recurrently in production (continuous training "using fresh data based on live pipeline triggers"); Level 2 adds automated build/test/package of pipeline components on commit and continuous delivery of pipeline implementations, with semi-automated pre-production deployment and manual production promotion — [Google Cloud](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- Retraining triggers: on schedule, on availability of new data, on model performance degradation, on concept drift detection — [Google Cloud](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning); Microsoft Level 4: "drift or regression signals trigger automatic retraining," with production metrics feeding retraining and model promotion "policy-based and automated" — [Microsoft Learn: MLOps maturity model](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model)
- Progressive-delivery patterns for models: CD4ML names **shadow models** (new version deployed alongside the current production model, receiving traffic without serving users) and **A/B testing** (competing models evaluated statistically); Microsoft Level 3 integrates "A/B testing of model performance" into deployment — [martinfowler.com: CD4ML](https://martinfowler.com/articles/cd4ml.html); [Microsoft Learn](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model)
- Champion/challenger via registry: MLflow documents the `@champion` alias pattern — a mutable alias designates the serving version, and promotion = reassigning the alias to the challenger version after validation — [MLflow Model Registry docs](https://mlflow.org/docs/latest/ml/model-registry/)
- CD4ML's three model-serving/release patterns — embedded model (model packaged in the application artifact, released through normal CD), model as a separate service, model-as-data ingested at runtime — determine which CI/CD pipeline a model release flows through — [martinfowler.com: CD4ML](https://martinfowler.com/articles/cd4ml.html)
- Everything-as-code precondition: Microsoft Level 1 ("DevOps but no MLOps") requires automated builds, automated application tests, and version-controlled code before any ML-specific automation is layered on — [Microsoft Learn](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model)

### Inferences
- "GitOps for models" in practice = the DVC/CD4ML arrangement where Git is the single source of truth for code + data pointers + pipeline definitions, and any environment state is reproducible from a commit (`git checkout` + `dvc checkout`/`dvc repro`); the fetched sources describe this mechanism without using the GitOps label.
- The maturity models agree that human approval remains in the loop at production promotion until the highest level (Google L2: "manual production deployment"; Microsoft L4 removes it via policy-based promotion) — manual gates are a feature of intermediate maturity, not a deficiency.

### Gaps
- No fetched authoritative source uses the term "GitOps for models" explicitly; the concept is an industry synthesis rather than a canonically named pattern in Google/Microsoft/CD4ML documents.

## Key Question 5: How do these patterns map onto common tools (MLflow, DVC, Kedro, W&B, Great Expectations)?

### Takeaway
Each major tool is the reference implementation of one canonical pattern: MLflow → experiment tracking + model registry/alias promotion; DVC → Git-plus-remote-storage data versioning and reproducible pipelines; Kedro → DAG pipelines, Data Catalog I/O abstraction, and standardized project structure; Great Expectations → declarative data validation with Checkpoints; W&B occupies the same experiment-tracking slot as MLflow Tracking.

### Cited Findings

| Tool | Pattern implemented | Canonical source |
|---|---|---|
| MLflow Tracking + Model Registry | Experiment tracking (experiments → runs → params/metrics/artifacts, lineage from version to producing run); registered models, versions, mutable aliases (`@champion`), tags for promotion workflows | [MLflow Model Registry docs](https://mlflow.org/docs/latest/ml/model-registry/) |
| DVC | Data/model versioning (`.dvc` metafiles in Git + content-addressed cache + remote storage; `dvc push/pull`, `git checkout`+`dvc checkout`); reproducible pipelines (`dvc.yaml`, `dvc repro`) | [DVC docs](https://doc.dvc.org/use-cases/versioning-data-and-models); named in [CD4ML](https://martinfowler.com/articles/cd4ml.html) as the reproducible-training tool |
| Kedro | DAG pipeline orchestration (nodes = pure functions with named I/O, dependency-resolved execution), Data Catalog (declarative dataset registry), standardized project template (conf/src/data/tests/notebooks separation) | [Kedro concepts docs](https://docs.kedro.org/en/stable/getting-started/kedro_concepts/) |
| Great Expectations | Declarative data validation (Expectations/Suites as "unit tests for data", Checkpoints + Actions in production pipelines, Data Docs) | [GX overview docs](https://docs.greatexpectations.io/docs/core/introduction/gx_overview) |
| MLflow Tracking (and Git branches) | CD4ML's experiment-tracking component explicitly names MLflow Tracking | [CD4ML](https://martinfowler.com/articles/cd4ml.html) |
| GoCD / CD servers | CD4ML's deployment-pipeline orchestration with governance gates | [CD4ML](https://martinfowler.com/articles/cd4ml.html) |

- Framework-level mapping: Google's Level 1 components (feature store, metadata store, pipeline triggers, data/model validation) and Microsoft's level gates ("experiment results are tracked," "ML registries" for promotion, Event Grid for event-driven retraining) describe the same tool categories platform-agnostically — [Google Cloud](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning); [Microsoft Learn](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model)

### Inferences
- Weights & Biases implements the same experiment-tracking pattern as MLflow Tracking (runs with logged config/metrics/artifacts under projects) plus artifact versioning; its canonical documentation is docs.wandb.ai.
- Tool pairings follow the pattern split, not competition: a typical stack combines MLflow (tracking/registry) + DVC (data versioning) + Kedro or Airflow (orchestration) + Great Expectations or pandera (data validation) — each covering one of the five workflow patterns in Question 2.

### Gaps
- W&B documentation was not independently fetched in this pass, so its specific concept names (runs, sweeps, artifacts, model registry) are stated from general knowledge, not a cited page.
- pandera and Airflow/Prefect/Dagster docs were likewise not fetched; their pattern classification (schema validation in code; DAG orchestration) is an inference from the verified pattern definitions above.
