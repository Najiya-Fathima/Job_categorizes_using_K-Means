import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from apscheduler.schedulers.background import BackgroundScheduler
import joblib
import time
import re
from sklearn.pipeline import Pipeline
import numpy as np
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('job_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'user_skills': ["python", "sql", "machine learning", "data analysis", "pandas"],
    'default_keyword': "data science",
    'pages_to_scrape': 3,
    'pipeline_file': 'job_clustering_pipeline.joblib',
    'jobs_file': 'scraped_jobs.csv',
    'max_retries': 3,
    'request_delay': 2
}

def scrape_karkidi_jobs(keyword=None, pages=3):
    """Scrape jobs from Karkidi with improved error handling"""
    if keyword is None:
        keyword = CONFIG['default_keyword']
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for page in range(1, pages + 1):
        url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
        logger.info(f"Scraping page: {page} for keyword: {keyword}")
        
        for attempt in range(CONFIG['max_retries']):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                break
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for page {page}: {e}")
                if attempt == CONFIG['max_retries'] - 1:
                    logger.error(f"Failed to scrape page {page} after {CONFIG['max_retries']} attempts")
                    continue
                time.sleep(2 ** attempt)  # Exponential backoff
        else:
            continue  # Skip this page if all attempts failed

        try:
            soup = BeautifulSoup(response.content, "html.parser")
            
            # More flexible job block selection
            job_blocks = soup.find_all("div", class_="ads-details")
            if not job_blocks:
                # Try alternative selectors
                job_blocks = soup.find_all("div", class_=re.compile(r"job|ad"))
            
            logger.info(f"Found {len(job_blocks)} job blocks on page {page}")
            
            for job in job_blocks:
                try:
                    # More robust data extraction with fallbacks
                    title_elem = job.find("h4") or job.find("h3") or job.find("h5")
                    title = title_elem.get_text(strip=True) if title_elem else "N/A"
                    
                    company_elem = job.find("a", href=lambda x: x and "Employer-Profile" in x) if job.find("a", href=lambda x: x and "Employer-Profile" in x) else job.find("a")
                    company = company_elem.get_text(strip=True) if company_elem else "N/A"
                    
                    location_elem = job.find("p")
                    location = location_elem.get_text(strip=True) if location_elem else "N/A"
                    
                    exp_elem = job.find("p", class_="emp-exp")
                    experience = exp_elem.get_text(strip=True) if exp_elem else "N/A"
                    
                    # Extract skills with multiple approaches
                    skills = ""
                    key_skills_tag = job.find("span", string="Key Skills")
                    if key_skills_tag:
                        skills_elem = key_skills_tag.find_next("p")
                        skills = skills_elem.get_text(strip=True) if skills_elem else ""
                    
                    # If no skills found, try alternative approaches
                    if not skills:
                        skills_keywords = ["skills", "requirements", "technologies"]
                        for keyword in skills_keywords:
                            skills_tag = job.find(string=re.compile(keyword, re.I))
                            if skills_tag:
                                skills_elem = skills_tag.find_next("p") if hasattr(skills_tag, 'find_next') else skills_tag.parent.find_next("p")
                                if skills_elem:
                                    skills = skills_elem.get_text(strip=True)
                                    break
                    
                    # Extract summary
                    summary = ""
                    summary_tag = job.find("span", string="Summary")
                    if summary_tag:
                        summary_elem = summary_tag.find_next("p")
                        summary = summary_elem.get_text(strip=True) if summary_elem else ""

                    # Only add jobs with meaningful data
                    if title != "N/A" and company != "N/A":
                        jobs_list.append({
                            "Title": title,
                            "Company": company,
                            "Location": location,
                            "Experience": experience,
                            "Summary": summary,
                            "Skills": skills,
                            "Scraped_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                except Exception as e:
                    logger.warning(f"Error parsing job block: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error processing page {page}: {e}")
            continue

        time.sleep(CONFIG['request_delay'])  # Be respectful to the server

    logger.info(f"Successfully scraped {len(jobs_list)} jobs")
    return pd.DataFrame(jobs_list)

def preprocess_skills(skills_series):
    """Enhanced preprocessing for skills data"""
    skills_series = skills_series.astype(str).fillna('')
    skills_series = skills_series.str.lower()
    skills_series = skills_series.str.strip()
    
    # More comprehensive cleaning
    skills_series = skills_series.str.replace(r'[;,/\(\)\[\]{}]', ' ', regex=True)
    skills_series = skills_series.str.replace(r'[^a-z0-9\s\-\+\#]', '', regex=True)
    skills_series = skills_series.str.replace(r'\s+', ' ', regex=True)
    skills_series = skills_series.str.strip()
    
    return skills_series

def find_best_k_using_silhouette(data, k_min=2, k_max=10):
    """Find optimal number of clusters with improved validation"""
    if len(data) < k_min:
        logger.warning(f"Not enough data points ({len(data)}) for clustering. Using k=2")
        k_min = k_max = 2
    
    try:
        tfidf = TfidfVectorizer(
            min_df=max(1, min(2, len(data)//10)), 
            max_df=0.85, 
            ngram_range=(1, 2),
            max_features=1000
        )
        X = tfidf.fit_transform(data)
        
        if X.shape[0] < k_min:
            logger.warning(f"TF-IDF resulted in {X.shape[0]} samples. Using k=2")
            return 2, tfidf
        
        best_score = -1
        best_k = k_min
        k_max = min(k_max, X.shape[0] - 1)  # Ensure k < n_samples

        for k in range(k_min, k_max + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                score = silhouette_score(X, labels)
                logger.info(f"k={k}, silhouette_score={score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                logger.warning(f"Error computing silhouette score for k={k}: {e}")
                continue

        logger.info(f"Best k based on silhouette score: {best_k} (score = {best_score:.4f})")
        return best_k, tfidf
        
    except Exception as e:
        logger.error(f"Error in clustering optimization: {e}")
        # Fallback to simple TF-IDF and k=3
        tfidf = TfidfVectorizer(max_features=100)
        return 3, tfidf

def train_final_pipeline(data, tfidf_vectorizer, k):
    """Train the final clustering pipeline"""
    try:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        pipeline = Pipeline([
            ('tfidf', tfidf_vectorizer),
            ('kmeans', kmeans)
        ])
        pipeline.fit(data)
        logger.info(f"Pipeline trained successfully with {k} clusters")
        return pipeline
    except Exception as e:
        logger.error(f"Error training pipeline: {e}")
        return None

def match_jobs_to_user(df, pipeline, user_skills=None):
    """Match jobs to user preferences with improved scoring"""
    if user_skills is None:
        user_skills = CONFIG['user_skills']
    
    if pipeline is None or df.empty:
        logger.warning("Cannot match jobs: pipeline is None or dataframe is empty")
        return
    
    try:
        vectorizer = pipeline.named_steps['tfidf']
        tfidf_matrix = vectorizer.transform(df['Skills_processed'])
        user_vec = vectorizer.transform([" ".join(user_skills)])

        # Calculate cosine similarity
        cosine_sim = (tfidf_matrix @ user_vec.T).toarray().flatten()
        
        # Get top matches (filter out zero similarity scores)
        valid_matches = cosine_sim > 0
        if not valid_matches.any():
            logger.info("No jobs found matching user skills")
            return
        
        valid_df = df[valid_matches]
        valid_scores = cosine_sim[valid_matches]
        
        top_indices = np.argsort(-valid_scores)[:5]
        top_matches = valid_df.iloc[top_indices]
        top_scores = valid_scores[top_indices]

        logger.info("\n--- Top Matched Jobs for You ---")
        for i, (idx, row) in enumerate(top_matches.iterrows()):
            logger.info(f"""
Job {i+1}:
Title: {row['Title']}
Company: {row['Company']}
Location: {row['Location']}
Experience: {row['Experience']}
Skills: {row['Skills'][:100]}...
Similarity Score: {top_scores[i]:.4f}
{'-'*50}""")
            
    except Exception as e:
        logger.error(f"Error matching jobs to user: {e}")

def save_jobs_data(df, filename=None):
    """Save scraped jobs to CSV"""
    if filename is None:
        filename = CONFIG['jobs_file']
    
    try:
        # If file exists, append new jobs (avoid duplicates)
        if os.path.exists(filename):
            existing_df = pd.read_csv(filename)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            # Remove duplicates based on title and company
            combined_df = combined_df.drop_duplicates(subset=['Title', 'Company'], keep='last')
            combined_df.to_csv(filename, index=False)
            logger.info(f"Appended {len(df)} jobs to existing file. Total: {len(combined_df)} jobs")
        else:
            df.to_csv(filename, index=False)
            logger.info(f"Saved {len(df)} jobs to {filename}")
    except Exception as e:
        logger.error(f"Error saving jobs data: {e}")

def load_pipeline():
    """Load existing pipeline if available"""
    try:
        if os.path.exists(CONFIG['pipeline_file']):
            pipeline = joblib.load(CONFIG['pipeline_file'])
            logger.info("Loaded existing pipeline")
            return pipeline
    except Exception as e:
        logger.warning(f"Could not load existing pipeline: {e}")
    return None

def job_pipeline(keyword=None):
    """Main job processing pipeline - now non-interactive"""
    try:
        logger.info("Starting job pipeline...")
        
        # Use provided keyword or default
        if keyword is None:
            keyword = CONFIG['default_keyword']
        
        # Scrape jobs
        df_jobs = scrape_karkidi_jobs(keyword, pages=CONFIG['pages_to_scrape'])

        if df_jobs.empty:
            logger.warning("No jobs scraped. Exiting pipeline.")
            return

        # Save scraped data
        save_jobs_data(df_jobs)

        # Preprocess skills
        df_jobs['Skills_processed'] = preprocess_skills(df_jobs['Skills'])
        df_jobs_clust = df_jobs[df_jobs['Skills_processed'].str.len() > 0]

        if df_jobs_clust.empty:
            logger.warning("No jobs with valid skills data for clustering")
            return

        # Try to load existing pipeline or create new one
        pipeline = load_pipeline()
        
        if pipeline is None:
            # Create new pipeline
            logger.info("Creating new clustering pipeline...")
            best_k, tfidf_vectorizer = find_best_k_using_silhouette(df_jobs_clust['Skills_processed'])
            pipeline = train_final_pipeline(df_jobs_clust['Skills_processed'], tfidf_vectorizer, best_k)
            
            if pipeline is not None:
                # Save the pipeline
                joblib.dump(pipeline, CONFIG['pipeline_file'])
                logger.info(f"Pipeline saved to {CONFIG['pipeline_file']}")
        
        if pipeline is not None:
            # Add cluster labels
            try:
                df_jobs_clust['Cluster_Label'] = pipeline.predict(df_jobs_clust['Skills_processed'])
            except Exception as e:
                logger.warning(f"Could not add cluster labels: {e}")

            # Match jobs to user
            match_jobs_to_user(df_jobs_clust, pipeline)
        
        logger.info("Job pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in job pipeline: {e}")

def run_interactive_mode():
    """Run the system interactively for testing"""
    print("=== Interactive Job Scraper ===")
    keyword = input("Enter the keyword (or press Enter for 'data science'): ").strip()
    if not keyword:
        keyword = CONFIG['default_keyword']
    
    job_pipeline(keyword)

def main():
    """Main function with improved scheduler setup"""
    print("=== Job Alert System ===")
    print("1. Run once interactively")
    print("2. Start automated daily scraping")
    
    try:
        choice = input("Choose option (1 or 2): ").strip()
        
        if choice == "1":
            run_interactive_mode()
        elif choice == "2":
            # Start scheduler
            scheduler = BackgroundScheduler(daemon=True)
            
            # Run immediately, then every 24 hours
            scheduler.add_job(
                job_pipeline, 
                'interval', 
                hours=24,
                id='job_scraping',
                replace_existing=True
            )
            
            # Run once immediately
            logger.info("Running initial job scraping...")
            job_pipeline()
            
            scheduler.start()
            logger.info("Automated job scraping started. Running every 24 hours.")
            logger.info("Press Ctrl+C to stop.")
            
            try:
                while True:
                    time.sleep(60)
            except (KeyboardInterrupt, SystemExit):
                logger.info("Shutting down scheduler...")
                scheduler.shutdown()
                logger.info("Scheduler stopped.")
        else:
            print("Invalid choice. Exiting.")
            
    except KeyboardInterrupt:
        logger.info("Program interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()