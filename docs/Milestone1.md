# Milestone 1: API Discovery, Data Ingestion, and Preprocessing

This milestone involves identifying a data source via a public API or an ethical web scraping/crawling target, writing a script to extract the data, performing initial preprocessing, and beginning exploratory analysis. The emphasis is on learning real-world data acquisition (handling REST endpoints, JSON structures, or parsing HTML/DOM trees) and choosing a source with enough features and records to enable meaningful data mining. This data source will potentially be what you perform your final data mining techniques on.

The submission should be a PDF file that contains the endpoints, target URLs, or source links to two potential data sources, a discussion of why the data is interesting to you, a sample of the extracted data, and the potential analysis you plan to perform.
Week 1: Research, Discovery, and Feasibility

    Explore either public, free APIs (e.g., World Bank, The Movie Database, OpenFDA, REST Countries) OR target websites suitable for web scraping/crawling (e.g., e-commerce catalogs, job boards, real estate listings, news archives, public forums).

    Identify at least two potential data sources that can yield a significant number of records and attributes (avoiding tiny, toy-scale responses or single-page scrapes).

    For Scraping/Crawling: Check the site's robots.txt and terms of service to ensure ethical, permissible data collection, and evaluate the HTML structure/pagination scheme.

    Consider the domain relevance of each source and how its structure can support a substantive data mining project.

## Deliverables (to include in your submission):

    Document listing potential APIs or target URLs, documentation/structure notes, and domain relevance.

    Justification for how the source can yield a rich dataset (e.g., proving pagination or crawling depth to reach over 10,000 records with 15+ attributes).

## Week 2: Extraction, Parsing, and Initial Exploration

    Discuss your top data source choice(s) and project ideas with the instructor to refine your approach.

    Write a preliminary extraction script (e.g., using Python with requests, BeautifulSoup, Selenium, Playwright, or Scrapy for scraping/crawling, or pandas/requests for APIs) to pull a sample of the data.

    Begin preliminary exploration: check data size after parsing, structure, feature types, and any ingestion challenges (e.g., rate limits, dynamic JavaScript rendering, missing fields, anti-scraping blocks).

Deliverables (to include in your submission):

    Summary of discussion with the instructor (feedback + data source decision).

    Documentation of the initial data pull (e.g., sample rows, descriptive statistics, structure of the parsed JSON or structured data frame).

## Week 3: Data Cleaning, Preliminary Analysis, and Report

    Perform preprocessing such as handling missing values, removing duplicates, cleaning text/HTML artifacts, and formatting inconsistent data types.

    Generate preliminary statistics: feature distributions, correlations, and missing data analysis.

Deliverables (Submit a final report):

    Access links, API endpoints, or target URLs (along with a code snippet or description of how the data was scraped/pulled).

    Description of the Data: Summary of records, attributes, data types, and early insights after ingestion.

    Project Ideas: Proposed directions for data mining analysis with the selected data source.

