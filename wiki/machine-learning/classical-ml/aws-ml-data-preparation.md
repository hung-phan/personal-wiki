---
name: aws-ml-data-preparation
description: Preparing data for machine learning on AWS — structured vs unstructured and labelled vs unlabelled data, storing datasets in S3, ETL with AWS Glue Studio (visual job joining S3 and a relational source into Redshift), no-code cleaning with Glue DataBrew (one-hot encoding, mean imputation, label encoding, publishing recipes), SageMaker Canvas Data Wrangler flows (encode, impute, standard scaler, PCA 5 → 3), feature engineering and scaling recap, cleanup order (Canvas app → user profile → domain → S3 → IAM roles) and cost warnings; how each console step maps to the scikit-learn pipeline. Study guide for Machine Learning A-Z, Part 11 (section 45).
created: 2026-11-13
updated: 2026-11-13
tags: [machine-learning, aws, s3, aws-glue, glue-databrew, sagemaker, data-wrangler, sagemaker-canvas, etl, preprocessing, study-guide, ml-a-z]
---
# Data Preparation on AWS: S3, Glue, DataBrew, Data Wrangler

> **How to read this note.** Study guide for *Machine Learning A-Z*, Part 11 (section 45, 12 items; aimed at an AWS machine-learning certification). Everything is done in the AWS console — no code — so this note maps each console step to its code equivalent in [data-preprocessing](data-preprocessing.md) instead of adding snippets. The "long version" Glue lab is an external recording and isn't covered. Before any hands-on, mind the cost: SageMaker domains and Canvas sessions bill while they exist.

## Why This Exists

**Problem.** The preprocessing from Part 1 runs on a laptop for a 10-row CSV. Real pipelines start from files in object storage and tables in databases, run on a schedule, and are built by people who may not write Python.

**Key insight.** AWS offers the same steps — store, join, clean, encode, impute, scale, reduce — at three levels of control: **Glue** (code or visual ETL jobs at scale), **Glue DataBrew** (point-and-click recipes for analysts), and **SageMaker Data Wrangler inside Canvas** (ML-oriented transforms that feed straight into model building).

**Reach for this when:**
- Your data lives in S3, Redshift or relational databases and must be prepared repeatedly.
- Non-programmers need to build cleaning steps.
- Preparing for an AWS ML certification.

**Don't reach for this when:**
- The dataset fits in memory and you're the only user — pandas + scikit-learn is simpler and free.
- You need custom feature logic that console transforms don't offer — write a Glue (Spark) or SageMaker Processing job.

## Diagrams

![The course's AWS parts as one lifecycle: prepare (S3, Glue, DataBrew, Data Wrangler), develop (Canvas, XGBoost, Model Registry, AI services), deploy (endpoints, containers, Lambda, IaC), automate (CodePipeline, SageMaker Pipelines), monitor (Model Monitor, Clarify, CloudWatch, budgets), with a retrain loop; inference options and cost habits](assets/aws-ml-lifecycle.svg)

## 1. Data types and storage

**Structured** data has rows and columns (CSV, database tables, JSON with a fixed schema); **unstructured** is text, images, audio. **Labelled** data has the target you want to predict; **unlabelled** data is used for clustering or pre-training. (The lecture's title also lists file formats — Parquet, ORC, Avro, RecordIO — which it doesn't discuss: columnar Parquet/ORC are the usual choice for analytics on S3; RecordIO is SageMaker's streaming format for built-in algorithms.)

**S3** is where datasets live: create a bucket (names are globally unique, so the course appends random characters), upload the CSV, and point every other service at its path.

## 2. ETL with Glue

**Glue Studio** builds jobs visually: source nodes (an S3 file, a relational table), transforms (join, filter, map columns), and a target (the course: Redshift). Glue runs the job on managed Spark and can be scheduled or triggered. The course's demo builds the graph without running it; Glue crawlers and the Data Catalog (which make S3 files queryable as tables) come up in the long lab.

## 3. No-code cleaning: DataBrew and Data Wrangler

The course cleans Part 1's `data.csv` (country, age, salary, purchased) twice:

| Step | Glue DataBrew | Canvas Data Wrangler | scikit-learn equivalent |
|---|---|---|---|
| Load | project + dataset from S3, IAM role | import from S3 into a data flow | `pd.read_csv` |
| Categories | one-hot encode Country, drop original | one-hot encode (as columns) | `OneHotEncoder` |
| Missing values | fill Age, Salary with the mean | impute mean | `SimpleImputer("mean")` |
| Target | map Yes/No to 1/0 | ordinal-encode Purchased | `LabelEncoder` |
| Scaling | — | standard scaler on Age, Salary | `StandardScaler` |
| Reduction | — | PCA, 5 features → 3 | `PCA(3)` |
| Output | publish the **recipe** (reusable step list), run a job | export / use the flow for Canvas models | a fitted `Pipeline` |

A DataBrew recipe or a Data Wrangler flow is the console version of a fitted pipeline: the same steps reapplied to new data. The same leakage rule applies — compute imputation means and scaler statistics from training data ([data-preprocessing](data-preprocessing.md) §4); console tools that transform the whole dataset at once don't do that for you.

**Feature engineering** is reviewed with the Part 1 income/age example: scale features so distances and gradients treat them comparably.

## 4. Cleaning up

Order matters, because resources depend on each other: delete the **Canvas app** (it bills per session hour), then the **user profile**, then the **SageMaker domain**, then empty and delete the **S3 buckets**, then the **IAM roles** created for the services. A forgotten domain or app is the most common surprise bill.

## Trade-offs

- **Glue vs DataBrew vs Data Wrangler.** Scale and full control (Spark) vs analyst-friendly recipes vs ML-specific transforms tied to SageMaker.
- **Console vs code.** Fast to start, visual vs versionable, testable, reviewable.
- **Managed vs local.** Scales and schedules vs cost and setup overhead for small data.

## Common Pitfalls

- **Leaving Canvas or a domain running.** They bill until deleted.
- **Transforming before splitting.** Console tools fit on everything you give them.
- **One-hot encoding high-cardinality columns in the console.** Hundreds of sparse columns; consider target encoding ([aws-ml-model-development](aws-ml-model-development.md) §3).
- **Deleting resources out of order.** Dependencies block deletion or leave orphans.

## Decision Table

| Scenario | Use this | Not that | Why |
|----------|----------|----------|-----|
| Scheduled joins across S3 and databases | Glue job | Notebook | Managed, repeatable |
| Analyst-built cleaning recipe | Glue DataBrew | Spark code | No code |
| Preparing features for SageMaker models | Data Wrangler / SageMaker Processing | DataBrew | Integrated with training |
| Small one-off dataset | pandas + scikit-learn | Any managed service | Simpler, free |

## References

- [AWS Glue — what is Glue](https://docs.aws.amazon.com/glue/latest/dg/what-is-glue.html) · [Glue DataBrew](https://docs.aws.amazon.com/databrew/latest/dg/what-is.html)
- [SageMaker Canvas — data preparation (Data Wrangler)](https://docs.aws.amazon.com/sagemaker/latest/dg/canvas-data-prep.html)
- [Amazon S3 user guide](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html)

## See Also

- [data-preprocessing](data-preprocessing.md) — Part 1, the same steps in code.
- [aws-ml-model-development](aws-ml-model-development.md) — Part 12.
- [aws-ml-deployment-mlops](aws-ml-deployment-mlops.md) — Parts 13–15.
