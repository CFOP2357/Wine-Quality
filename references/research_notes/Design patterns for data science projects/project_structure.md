# Project Structure and Repository Layout Patterns for Data Science Projects

## What is Cookiecutter Data Science (ccds), what structure does it prescribe, and what principles does it state?

### Takeaway
Cookiecutter Data Science (by DrivenData) is the de facto standard template for data science repositories (~10.1k GitHub stars, 2.6k forks). It prescribes a staged `data/` directory (external/interim/processed/raw), a Python source package (named after the project in v2, replacing the generic `src/` of v1), plus `notebooks/`, `models/`, `reports/`, `references/`, and `docs/` — grounded in explicit opinions: raw data is immutable, analysis is a DAG, notebooks are for exploration while source code is for reproduction, and data stays out of version control.

### Cited Findings
- ccds describes itself as "A logical, reasonably standardized but flexible project structure for doing and sharing data science work"; the repo has ~10.1k stars and ~2.6k forks (as of Oct 2026) — [GitHub: drivendataorg/cookiecutter-data-science](https://github.com/drivendataorg/cookiecutter-data-science)
- v2 requires installing the `cookiecutter-data-science` Python package (recommended via `pipx install cookiecutter-data-science`) and using the `ccds` CLI instead of generic `cookiecutter` — [GitHub README](https://github.com/drivendataorg/cookiecutter-data-science); [ccds homepage](https://cookiecutter-data-science.drivendata.org/)
- Prescribed directory tree (v2): `LICENSE`, `Makefile`, `README.md`, `data/` (`external` = third-party sources, `interim` = intermediate transformed data, `processed` = final datasets for modeling, `raw` = "original immutable data dump"), `docs/` (default mkdocs project), `models/` (trained models and predictions), `notebooks/`, `pyproject.toml`, `references/` (data dictionaries, manuals), `reports/` + `reports/figures/`, `requirements.txt`, `setup.cfg`, and a source package `[module_name]/` containing `__init__.py`, `config.py`, `dataset.py`, `features.py`, `modeling/train.py`, `modeling/predict.py`, `plots.py` — [ccds homepage](https://cookiecutter-data-science.drivendata.org/)
- v2 change: "The source directory structure now uses the actual module name rather than a generic 'src' folder"; project creation now offers a choice of environment managers (virtualenv, conda, pipenv, uv, pixi, poetry), testing frameworks, and linting/formatting tools — [ccds homepage](https://cookiecutter-data-science.drivendata.org/)
- Stated opinion: "treat your data analysis pipeline as a directed acyclic graph (DAG)" — anyone should be able to re-run the analysis from code plus raw data and get identical outputs — [ccds Opinions](https://cookiecutter-data-science.drivendata.org/opinions/)
- Stated opinion: "raw data must be treated as immutable" — read and copy it, never edit it in place (especially not manually in Excel), never overwrite it with processed versions, never keep multiple raw versions; data flows `raw/` → `interim/` → `processed/` — [ccds Opinions](https://cookiecutter-data-science.drivendata.org/opinions/)
- Stated opinion: data stays out of source control — "by default, the `data/` folder is included in the `.gitignore` file"; large data goes to cloud storage (S3, Azure Blob, GCS) synced with tools like `awscli`, `azcopy`, or `cloudpathlib` — [ccds Opinions](https://cookiecutter-data-science.drivendata.org/opinions/)
- Stated opinion: notebooks are for exploration; "Source code is superior for replicability because it is more portable, can be tested more easily, and is easier to code review" — [ccds Opinions](https://cookiecutter-data-science.drivendata.org/opinions/)
- Stated opinion: "build from the environment up" — reproducibility needs reproducible computational environments; conda (via Miniforge) preferred because it handles non-Python dependencies; Docker/Vagrant for complex cases — [ccds Opinions](https://cookiecutter-data-science.drivendata.org/opinions/)
- Stated opinion: keep secrets/config out of version control — use a git-ignored `.env` loaded with `python-dotenv`; AWS credentials via `~/.aws/credentials` profiles — [ccds Opinions](https://cookiecutter-data-science.drivendata.org/opinions/)
- Stated meta-opinion on adaptation: "be liberal in changing the folders around for _your_ project, but be conservative in modifying the default cookiecutter structure for _all_ projects" — [ccds Opinions](https://cookiecutter-data-science.drivendata.org/opinions/)
- Rationale for standardization: a well-defined standard structure means newcomers can understand an analysis without extensive documentation and without reading all the code first — [ccds "Why"](https://cookiecutter-data-science.drivendata.org/why/); summarized by [KDnuggets](https://www.kdnuggets.com/2018/07/cookiecutter-data-science-organize-data-project.html)

### Inferences
- ccds is best understood as a set of conventions + rationale rather than tooling: the folders encode the DAG opinion (raw → interim → processed → models → reports) in the filesystem itself.
- The v2 rename of `src/` to the project's module name aligns ccds with standard Python packaging (an importable, installable package), signaling that the template expects pipeline code to be a real package, not loose scripts.

### Gaps
- No official statistic exists on how many projects use ccds; GitHub stars/forks are the only adoption proxy the project publishes.

## What alternative structure templates exist (Kedro, TDSP, DagsHub/DVC, MLOps accelerators, nbdev)?

### Takeaway
The main currently maintained alternatives are Kedro's project template (conf/data/src with a pipeline framework and a numbered 8-layer data convention), DagsHub's DVC-oriented cookiecutters, Azure's MLOps v2 accelerator (a multi-repo "project factory"), and nbdev (notebook-first literate programming). Microsoft's TDSP template (Code/Docs/Sample_Data) is historically influential but its GitHub repo was archived in October 2023.

### Cited Findings
- **Kedro**: the standard project includes `conf/` (split into `base` for shared settings and `local` for personal credentials), `src/` (pipelines and nodes), `data/`, `tests/`, `notebooks/`, `docs/`; `conf/base` holds `catalog.yml` (Data Catalog), `parameters.yml`, and optional `logging.yml`; docs recommend keeping `pipeline_registry.py` and `settings.py` in default locations — [Kedro concepts docs](https://docs.kedro.org/en/stable/get_started/kedro_concepts/)
- **Kedro data layers**: Kedro popularized a numbered data-engineering convention inside `data/`: `01_raw` (raw immutable data, "single source of truth"), `02_intermediate` (typed data), `03_primary` (domain model), `04_feature` (model features), `05_model_input` ("master tables"), `06_models` (serialized models), `07_model_output`, `08_reporting` (ad hoc descriptive cuts); the convention is credited to Bruce Philp and Guilherme Braccialli and is meant so "everyone has an idea of which data cleaning or processing steps have happened" — [Kedro FAQ/glossary](https://docs.kedro.org/en/stable/getting-started/glossary/); [TDS: The importance of layered thinking in data engineering](https://towardsdatascience.com/the-importance-of-layered-thinking-in-data-engineering-a09f685edc71)
- **TDSP (Microsoft Team Data Science Process)**: "an agile, iterative, data science methodology" supported by "a lifecycle definition, standard project structure, artifact templates, and tools"; its project template has three top-level folders — `Code` (development/analysis scripts), `Docs` (document templates, notably the project Charter and Exit Report), and `Sample_Data` (explicitly "NOT supposed to contain LARGE raw or processed data") — [Azure-TDSP-ProjectTemplate](https://github.com/Azure/Azure-TDSP-ProjectTemplate)
- **TDSP maintenance status**: "This repository was archived by the owner on Oct 12, 2023. It is now read-only," though it retains ~579 stars / ~442 forks as a reference — [Azure-TDSP-ProjectTemplate](https://github.com/Azure/Azure-TDSP-ProjectTemplate)
- **DagsHub templates**: DagsHub offers "Create from template" project templates: a Cookiecutter-DVC template "based on the well-known Cookiecutter Data Science template, modified to give you a basic DVC pipeline right out of the box"; a Jupyter Notebook + DVC template with notebook cells for DVC versioning; and a Cookiecutter-MLOps template with data/models/visualization script folders plus `dvc.yaml` and `dvc.lock` defining the ML pipeline — [DagsHub project templates docs](https://dagshub.com/docs/feature_guide/project_templates/); [DagsHub Cookiecutter-MLOps](https://dagshub.com/roysh/Cookiecutter-MLOps)
- **Azure MLOps v2 accelerator**: a "project factory" of three repositories — a header `mlops-v2` repo that bootstraps projects, an `mlops-templates` repo deploying base CI/CD pipelines, and project-template repos deploying Azure ML resources per scenario; bootstrapping "encapsulates key MLOps decisions such as the components of the repository, the structure of the repository, the link between model development and model deployment, and technology choices"; the header is a mono-repo of configurable architectural patterns (e.g., Terraform or Bicep infra × GitHub Actions or Azure DevOps CI/CD) — [Azure/mlops-v2](https://github.com/Azure/mlops-v2); [ODSC: MLOps v2 Solution Accelerator](https://opendatascience.com/mlops-v2-solution-accelerator-unifying-mlops-at-microsoft/)
- **nbdev** (fast.ai, by Jeremy Howard and Hamel Husain): "allows you to fully develop a library in Jupyter notebooks, putting all your code, tests and documentation in one place"; `#export`-flagged cells are exported into Python modules, docs are auto-generated (via Quarto), and notebook cells run as unit tests; used to build fastai and fastcore — [InfoWorld nbdev v2 review](https://www.infoworld.com/article/3686516/nbdev-v2-review-git-friendly-jupyter-notebooks.html); [nbdev on PyPI](https://pypi.org/project/nbdev/); [Overstory: How nbdev helps us structure our DS workflow](https://overstory.com/blog/how-nbdev-helps-us-structure-our-data-science-workflow-in-jupyter-notebooks)
- **DVC-oriented layouts** generally center the repo on `dvc.yaml` (pipeline stages), `params.yaml`, and `.dvc` pointer files, with data/models stored in remote storage — reflected in DagsHub's templates above — [DagsHub project templates docs](https://dagshub.com/docs/feature_guide/project_templates/)

### Inferences
- Templates split into two families: *layout conventions* (ccds, TDSP, DagsHub cookiecutters) that only organize files, and *framework templates* (Kedro, Azure MLOps v2, nbdev) where the structure is enforced/consumed by tooling (data catalog, CI/CD generators, notebook exporter). Framework templates are stickier but impose more learning curve.
- TDSP's archival means Microsoft's actively promoted path is the MLOps v2 accelerator; TDSP survives mainly as methodology documentation and a historical reference.

### Gaps
- I could not fetch a page showing Kedro's literal full default tree with the numbered data subfolders in one place (the stable-docs tutorial page no longer prints the tree); the numbered layers are confirmed via Kedro's FAQ/glossary and the cited TDS article, but exact current starter contents (e.g., whether new `kedro new` starters still pre-create all eight numbered folders) were not verified first-hand.
- DVC's own recommended-layout doc redirected and was not fetched; DVC layout claims here rest on DagsHub's template docs.

## What are the conventions for notebooks vs. source packages?

### Takeaway
The consensus convention: notebooks are a lab for exploration and communication, while anything re-run or shared gets refactored into an importable source package; ccds prescribes ordered, authored notebook filenames (`<step>-<ghuser>-<description>.ipynb`), and Python packaging norms increasingly favor the "src layout" so the installed package — not the working tree — is what gets imported.

### Cited Findings
- ccds: use `notebooks/exploratory/` for initial work and `notebooks/reports/` for polished analyses; name notebooks `<step>-<ghuser>-<description>.ipynb` (number for ordering, author handle, short description); avoid collaborating directly in the same notebook because merge conflicts on JSON are painful; "refactor the good parts" of repeated notebook code into source modules and import them back into notebooks — [ccds Opinions](https://cookiecutter-data-science.drivendata.org/opinions/)
- ccds rationale: "Source code is superior for replicability because it is more portable, can be tested more easily, and is easier to code review" — [ccds Opinions](https://cookiecutter-data-science.drivendata.org/opinions/)
- Python packaging "src layout": importable code lives in `src/<package>/` rather than at repo root; it "requires installation of the project to be able to run its code" and "helps avoid" accidentally importing the development tree, because "the Python interpreter includes the current working directory as the first item on the import path"; editable installs of a flat layout can also leak config files (`setup.py`, `noxfile.py`) onto the import path — [Python Packaging User Guide: src layout vs flat layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
- ccds v2 uses a flat package named after the project (e.g., `my_project/` with `dataset.py`, `features.py`, `modeling/`) rather than a literal `src/` directory — [ccds homepage](https://cookiecutter-data-science.drivendata.org/)
- nbdev is the explicit counter-position: the notebook *is* the source of truth, and modules, tests, and docs are generated from it, "incentivizing clear code, proper Git version control, and continuous documentation and testing" — [InfoWorld nbdev v2 review](https://www.infoworld.com/article/3686516/nbdev-v2-review-git-friendly-jupyter-notebooks.html)
- Kedro's take: notebooks live in a `notebooks/` folder for experimentation, but production logic belongs in `src/` as pipeline nodes registered in `pipeline_registry.py` — [Kedro concepts docs](https://docs.kedro.org/en/stable/get_started/kedro_concepts/)

### Inferences
- "src layout" means two different things in the wild: the PyPA sense (a `src/` wrapper directory) and the loose DS sense (ccds v1's `src/` package). ccds v2's rename resolves the ambiguity by making the package importable under its real name — note that ccds v2 is technically a *flat* layout in PyPA terms, just with a properly named package.
- The notebook-naming convention (`01-abc-eda.ipynb`) operationalizes the DAG opinion at the notebook level: lexicographic order encodes execution order.

### Gaps
- No single authoritative source quantifies how widely the `<step>-<initials>-<description>` notebook naming convention is adopted beyond ccds's own docs; it is widely echoed in blog posts but unmeasured.

## How are data folders staged (raw/interim/processed/external) and why is raw data immutable?

### Takeaway
Both major conventions stage data as a one-way pipeline on disk — ccds with four folders (raw → interim → processed, plus external), Kedro with eight numbered layers — because treating raw data as an immutable single source of truth is what makes the whole analysis reproducible from code + raw inputs alone.

### Cited Findings
- ccds staging: `data/raw` = "original, immutable data dump"; `data/interim` = intermediate transformed data (also a cache for long-running steps); `data/processed` = final canonical datasets for modeling; `data/external` = third-party sources — [ccds homepage](https://cookiecutter-data-science.drivendata.org/)
- ccds rationale for immutability: if raw data is never modified, anyone can reproduce the final products with "only the code in `src` and the data in `data/raw`"; the don'ts are explicit — don't edit raw data manually ("especially in Excel"), don't overwrite it, don't save multiple versions of it — [ccds Opinions](https://cookiecutter-data-science.drivendata.org/opinions/)
- Because immutable data is reproducible from raw, ccds gitignores `data/` entirely and pushes storage/versioning to cloud object stores — [ccds Opinions](https://cookiecutter-data-science.drivendata.org/opinions/)
- Kedro's layered equivalent: `01_raw` "should never be changed, it forms your single source of truth"; `02_intermediate` exists to type raw data (e.g., parse strings); downstream layers (`03_primary`, `04_feature`, `05_model_input`, `06_models`, `07_model_output`, `08_reporting`) separate domain modeling, feature engineering, model artifacts, and reporting outputs — [TDS: layered thinking in data engineering](https://towardsdatascience.com/the-importance-of-layered-thinking-in-data-engineering-a09f685edc71); [Kedro glossary/FAQ](https://docs.kedro.org/en/stable/getting-started/glossary/)
- Stated collaboration benefit of layering: "everyone has an idea of which data cleaning or processing steps have happened" just from where a file sits — [TDS: layered thinking](https://towardsdatascience.com/the-importance-of-layered-thinking-in-data-engineering-a09f685edc71)
- TDSP's contrasting choice: its `Sample_Data` folder holds only small sample/test data, with real data kept outside the repo entirely — [Azure-TDSP-ProjectTemplate](https://github.com/Azure/Azure-TDSP-ProjectTemplate)

### Inferences
- Raw-data immutability is the filesystem analogue of the "analysis is a DAG" principle: mutating raw inputs would create a cycle (outputs feeding back into inputs), destroying reproducibility and cacheability of intermediate stages.
- Kedro's eight layers are a superset/refinement of ccds's four: raw≈01, interim≈02–05, processed≈05, models≈06, reports≈07–08; teams using DVC add content-addressed versioning on top so "immutable" is enforced by tooling rather than discipline.

### Gaps
- None significant; this is the best-documented area of the topic.

## What do practitioners criticize or vary in these templates?

### Takeaway
Common criticisms: ccds is over-engineered for small analyses (packaging scaffolding like setup files mixed into exploratory work), rigid structure can impede the iterative learning that experimentation requires, and the templates don't answer the repo-granularity question — where monorepos preserve relationships between many short analyses and ease code sharing, while per-project repos keep deployments independent.

### Cited Findings
- ccds "has been criticized for being overly complicated and mixing logic of building Python packages (like setup.py and LICENSE...) with data science," with practitioners questioning whether analysis functions really need to be pip-installable — [TDS: It's time to structure your data science project](https://towardsdatascience.com/its-time-to-structure-your-data-science-project-1fa064fbe46)
- Broader concern: "models, hyperparameters, and feature transformations need to be learned through experimentation and iteration," so "too much structure and process around data science projects can hinder on-the-go learning" — [Andrew Wheeler: Musings on Project Organization](https://andrewpwheeler.com/2021/09/09/musings-on-project-organization-books-and-courses/)
- ccds itself anticipates the criticism and sanctions variation: simplify unnecessary folders for small projects, expand for complex ones, reorganize congested directories — but keep the shared default conservative across projects — [ccds Opinions](https://cookiecutter-data-science.drivendata.org/opinions/)
- Monorepo vs multi-repo trade-offs for DS: monorepos make code sharing straightforward and centralize dependency management/versions, while multi-repos "require each repository to manage its dependencies independently, which can lead to version mismatches"; monorepo projects can still be built/tested/deployed independently — [DataCamp monorepo tutorial](https://www.datacamp.com/tutorial/monorepo); [Databricks community: monorepo vs multi-repo for asset bundles](https://community.databricks.com/t5/community-articles/monorepo-vs-multi-repo-for-databricks-asset-bundles-a-decision/td-p/165123)
- DS-specific monorepo argument: it "better preserves relationships between small (1–2 mo) analysis projects" — many DS efforts are too short-lived to justify a full per-project repo + template — [Posit forum discussion](https://forum.posit.co/t/multiple-rproj-in-a-single-github-repo/31927)
- Vendor forks as institutionalized variation: DagsHub ships a modified ccds with a DVC pipeline baked in, showing the common practice of extending ccds with data-versioning tooling rather than replacing it — [DagsHub project templates docs](https://dagshub.com/docs/feature_guide/project_templates/)
- nbdev adherents argue the notebook/source split is a false dichotomy: with export + tests + docs generated from notebooks, "serious software development" can happen notebook-first — [Overstory blog](https://overstory.com/blog/how-nbdev-helps-us-structure-our-data-science-workflow-in-jupyter-notebooks); [TDS: Serious Software Development Using Jupyter Notebooks](https://towardsdatascience.com/serious-software-development-using-jupyter-notebooks-95f84f4d8ef8)

### Inferences
- The criticism pattern mirrors template maturity: v1 complaints (generic `src/`, setup.py boilerplate) were partially addressed by ccds v2 (named package, pyproject.toml, choice of env/lint/test tooling), suggesting the maintainers treat criticisms as roadmap input.
- The practical consensus across critics and advocates converges on: adopt the shared skeleton for consistency, prune ruthlessly per project, and let tooling (DVC, Kedro catalog, Makefiles) enforce the conventions that discipline alone won't.

### Gaps
- I found no rigorous survey data (e.g., percentage of DS teams using ccds vs Kedro vs ad hoc structures); adoption comparisons rest on GitHub stars and anecdotal blog/forum evidence.
- Search results summarizing the TDS critique article and the Posit forum monorepo point came from search snippets rather than full-page fetches; the quoted claims match the snippets but were not verified against the full articles.
