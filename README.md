# ProjectCatalyst_Reputation_Scores_POC
https://projectcatalyst.io/funds/11/cardano-use-cases-concept/reputation-scores-for-catalyst-proposers-and-reviewers-by-lidonation-and-trustlevel

## Overview
Welcome to the Project Catalyst Reputation Scores POC repository. This project aims to conduct a comprehensive analysis of previous funding rounds (up to Fund 10) using data from Catalyst Testnet, Ideascale Platform, Milestone Reporting, Community Reviewing data, and more. The goal is to link reviews with voting results and proposal outcomes and develop a knowledge graph to identify quality reviewers and voters.

## Repository Structure
- `/data`: Contains the datasets used for analysis.
- `/scripts`: Python scripts for data processing and analysis.
- `/reports`: Detailed reports of findings.
- `/graphs`: Knowledge graph files and network analysis outputs.

# Milestone 1: Research and Data Analysis

## 1. Comparative Analysis Linking Reviews with Voting Results and Proposal Outcomes

### Key Findings
- **Fund 9:**
  - Correlation coefficient: 0.045 (very weak positive correlation).
- **Fund 8:**
  - Correlation coefficient: 0.06 (weak positive correlation).
- **Fund 10:**
  - Preliminary data shows a slightly negative correlation.

### Reviewer Accuracy
- Reviewers with an accuracy of 0.9 or higher showed significant correlation with project outcomes, making up about one-third of all reviewers.

### Implications
- Feasibility assessments alone are not sufficient to predict project success.
- A comprehensive evaluation system including additional metrics such as impact assessments is needed.

### Data Overview
- Public Spreadsheets:
  - vCA Aggregated - Fund 8
  - vPA Aggregate File - Fund 9
  - F10 Community Review - Aggregate File 
  - Catalyst Public Reporting Tracker

### Analysis Steps
1. Categorize the Feasibility Rating as:
   - Accurate Rating: Ratings of 4 and 5
   - Wrong Rating: Ratings of 1, 2, and 3
2. Merge data on Project Name to align outcomes with feasibility ratings.
3. Convert Project Status to binary format:
   - Completed = 1
   - Pending = 0
4. Determine correct predictions based on feasibility ratings.
5. Calculate the percentage of outcomes correctly predicted by each reviewer.
6. Create box plots to compare feasibility ratings by project outcome.

### Fund Analysis
- **Fund 8:**
  - 77.46% of reviewers predicted outcomes correctly at least 50% of the time.
  - Correlation coefficient: 0.0685 (very weak positive correlation).
- **Fund 9:**
  - Overall correlation: 0.0454 (very weak positive correlation).
  - For reviewers with high accuracy (≥ 0.9), correlation: ~0.499.
  - Mean accuracy: 70.19%.
- **Fund 10:**
  - Correlation with projects being completed on time: -0.046.
  - Correlation with projects being completed: -0.137.

## 2. Development of a Knowledge Graph and Network Analysis to Identify Quality Reviewers

### Ontology Structure
- **Entities (Nodes):**
  - Project: Attributes (`Project Title`, `Overall Feasibility Rating`, `Project Status`)
  - Reviewer: Attributes (`Reviewer ID`, `Reviewer Accuracy`)
- **Relationships (Edges):**
  - Reviewed: Connects a `Reviewer` to a `Project` with attributes (`Feasibility Rating`, `Rating Accuracy`).

### Steps to Create the Knowledge Graph
1. Create Nodes for projects and reviewers.
2. Create Edges connecting reviewers to projects reviewed.

### CSV File Structure (see 'data')
- **Project Nodes:**
  - `Project Title`, `Overall Feasibility Rating`, `Project Status`
- **Reviewer Nodes:**
  - `Reviewer ID`, `Reviewer Accuracy`
- **Reviewed Edges:**
  - `Reviewer ID`, `Project Title`, `Feasibility Rating`, `Rating Accuracy`

### Demonstration
- [Knowledge Graph Video](https://www.loom.com/share/43110018ae364a75b0b53fd1d49627f6)

### Network Analysis Insights
- High-Accuracy Reviewers: Identified reviewers with consistent high accuracy.
- Aggregation Across Rounds: Cross-fund evaluation helps establish long-term accuracy scores.

## Acknowledgments
Special thanks to all contributors and the Catalyst community for their valuable data and insights.
