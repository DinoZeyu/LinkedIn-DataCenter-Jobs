import time, re, random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException,
)
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys

from url_generator import FULL_FILTER_DEFINITIONS, generate_urls, FULL_FILTER_ORDER, extend_url_with_filter
from utils import extract_number_results, extract_job_data, simulate_human_like_actions

import os
from dotenv import load_dotenv


class CrawlerJob:
    # A job to crawl a specific URL with a specific handler function
    # url: Request URL
    # handler: A function to get contents from URLs, func(driver, url, time_sleep, wait_time) -> CrawlerResult
    def __init__(self, url, handler):
        self.url = url
        self.handler = handler
        self.attempts = 0
        # self.result = None  
        # self.error = None  
        # self.attempts = 0 
        # self.success = False  

class CrawlerResult:
    def __init__(self, url, data=None, crawler_type=None):
        self.url = url
        self.data = data  # data can be list[CrawlerJob] or list[dict]
        self.crawler_type = crawler_type  # choice of ['list', 'detail']


def result_router(result: CrawlerResult, job_queue, results, results_lock) -> None:
    if not result or not result.data:
        return
    # Depend on the type of result, either enqueue new jobs or save results
    if result.crawler_type == 'list':
        # Get new jobs and enqueue them
        # result.data is list[CrawlerJob]，each dict has keys: url, handler
        for job in result.data:
            job_queue.put(job)

    elif result.crawler_type == 'detail':
        with results_lock:
            results.append({"url": result.url, "jobs": result.data})

    else:
        print(f"Unknown Type: {result.crawler_type}")

def wait_get_element(driver, selector, timeout=10):
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        )
        return element
    except TimeoutException:
        print(f"Timeout: Element with selector '{selector}' not found within {timeout} seconds.")
        return None
    
def wait_for_element(driver, selector, timeout=10):
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        )
        return True
    except TimeoutException:
        print(f"Timeout: Element with selector '{selector}' not found within {timeout} seconds.")
        return False

def login_linkedin():
    load_dotenv()
    USERNAME = os.getenv("LINKEDIN_USERNAME")
    PASSWORD = os.getenv("LINKEDIN_PASSWORD")
    if not USERNAME or not PASSWORD:
        raise ValueError("Plz set LINKEDIN_USERNAME and LINKEDIN_PASSWORD in .env file")
        
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    # login to LinkedIn login page
    driver.get("https://www.linkedin.com/login")
    time.sleep(2)

    # Enter email and password
    driver.find_element(By.ID, "username").send_keys(USERNAME)
    driver.find_element(By.ID, "password").send_keys(PASSWORD + Keys.RETURN)
    time.sleep(5)

    # Verify login
    if "feed" in driver.current_url or "jobs" in driver.current_url:
        print("✅ Login successful!")
        return True, driver
    else:
        print("❌ Login may have failed.")
        return False, driver
    
def login_linkedin_driver(driver):
    load_dotenv()
    USERNAME = os.getenv("LINKEDIN_USERNAME")
    PASSWORD = os.getenv("LINKEDIN_PASSWORD")
    if not USERNAME or not PASSWORD:
        raise ValueError("Plz set LINKEDIN_USERNAME and LINKEDIN_PASSWORD in .env file")

    # login to LinkedIn login page
    driver.get("https://www.linkedin.com/login")
    time.sleep(2)

    # Enter email and password
    driver.find_element(By.ID, "username").send_keys(USERNAME)
    driver.find_element(By.ID, "password").send_keys(PASSWORD + Keys.RETURN)
    time.sleep(5)

    # Verify login
    if "feed" in driver.current_url or "jobs" in driver.current_url:
        print("✅ Login successful!")
        return True
    else:
        print("❌ Login may have failed.")
        return False
        
        
def linkedin_common_crawler(driver, url, time_sleep=1, wait_time=10):
    driver.get(url)
    time.sleep(time_sleep)

    job_main = wait_get_element(driver, "main#main", timeout=wait_time)
    if not job_main:
        return None

    # Wait for the first job card to load
    if not wait_for_element(driver, "ul:first-of-type>li.ember-view div.artdeco-entity-lockup__metadata", timeout=wait_time):
        return None
    
    job_card_list = job_main.find_elements(By.CSS_SELECTOR, "ul:first-of-type>li.ember-view")

    all_job_data = list(map(extract_job_data, job_card_list))

    return {
        "url": url,
        "jobs": all_job_data
    }

def get_linkedin_job_main_page(driver, url, time_sleep=1, wait_time=10, scroll=False, _refresh_attempt=0):
    # Get the main job listing container from a LinkedIn job search URL
    # If scroll is True, scroll through the job list to load more jobs
    previous_main = None
    try:
        previous_main = driver.find_element(By.CSS_SELECTOR, "main#main")
    except NoSuchElementException:
        previous_main = None

    timeout_exc = None
    try:
        driver.get(url)
    except TimeoutException as exc:
        timeout_exc = exc
        print(f"Loading Time Exceed: {url}")
        # force stop loading to prevent further delay
        try:
            driver.execute_script("window.stop();")
        except WebDriverException:
            pass
    except WebDriverException as exc:
        print(f"driver.get has problem: {url} ({exc})")
        raise RuntimeError(f"driver.get failed: {url}") from exc

    time.sleep(time_sleep + random.randint(0, 2))

    wait_timeout = wait_time * 2 if timeout_exc is not None else wait_time

    if timeout_exc is not None and previous_main is not None:
        try:
            WebDriverWait(driver, wait_timeout).until(EC.staleness_of(previous_main))
        except TimeoutException:
            if _refresh_attempt >= 1:
                print(f"Page Loading Failed and DOM not Refreshing: {url}")
                raise RuntimeError(f"Page Loading Failed and DOM not Refreshing: {url}") from timeout_exc
            print(f"Page Loading Failed and DOM not Refreshing, Try forcing refreshing: {url}")
            try:
                driver.execute_script("window.stop();")
            except WebDriverException:
                pass
            driver.refresh()
            time.sleep(random.uniform(1.0, 2.5))
            return get_linkedin_job_main_page(
                driver,
                url,
                time_sleep=time_sleep,
                wait_time=wait_time,
                scroll=scroll,
                _refresh_attempt=_refresh_attempt + 1,
            )

    simulate_human_like_actions(driver, 1, 2)

    job_main = wait_get_element(driver, "main#main", timeout=wait_timeout)
    if not job_main:
        if timeout_exc is not None:
            raise RuntimeError(f"Loading Time Exceed: {url}") from timeout_exc
        return None

    if scroll:
        # find scrollable job list container scaffold-layout__list>div
        scrollable = job_main.find_element(By.CSS_SELECTOR, "div.scaffold-layout__list>div")
        scroll_height = driver.execute_script("return arguments[0].scrollHeight", scrollable)
        position = 0
        step = 300

        while position < scroll_height:
            position += step
            driver.execute_script("arguments[0].scrollTo(0, arguments[1]);", scrollable, position)
            time.sleep(random.uniform(0.1, 0.4))
            scroll_height = driver.execute_script("return arguments[0].scrollHeight", scrollable)

    if timeout_exc is not None:
        print(f"页面加载超时但 DOM 已可用: {url}")

    return job_main


def linkedin_page_crawler(driver, url, time_sleep=1, wait_time=10) -> CrawlerResult:
    # Get the total number of jobs from a LinkedIn job search URL
    # If total jobs > 1000, generate refined filter tasks to queue, (url, linkedin_page_crawler)
    # If total jobs <= 1000, generate job detail tasks to queue, (url, linkedin_job_crawler)
    # Return: CrawlerResult {url, list[CrawlerJob], 'list'} or CrawlerResult {url, list[CrawlerJob], 'detail'}

    def generate_paged_urls(base_url, total_jobs):
        paged_urls = []
        for start in range(0, min(1000 - 25, total_jobs), 25): # LinkedIn 最多只允许翻到第 1000 条
            paged_url = f"{base_url}&start={start}"
            paged_urls.append(paged_url)
        return paged_urls

    print(f"Access Page: {url}")
    job_main = get_linkedin_job_main_page(driver, url, time_sleep, wait_time)
    if not job_main:
        return CrawlerResult(url, [], 'list')
    total_jobs = extract_number_results(job_main)
    if total_jobs is None:
        print("Unable to extract total job count. Skip this URL.")
        return CrawlerResult(url, [], 'list')
    print(f"Total jobs found: {total_jobs}")

    jobs = []

    if total_jobs > 1000:
        print("More than 1000 jobs, generating refined filter tasks...")
        current_filters = re.findall(r"f_[^&=]+", url)
        existing_filter_keys = set(current_filters)
        available_filters = [(f[0], f[1].param_key) for f in FULL_FILTER_DEFINITIONS.items() if f[1].param_key not in existing_filter_keys]
        if not available_filters:
            print("No available filters left, generating job detail tasks...")
            jobs.extend(CrawlerJob(url, linkedin_job_crawler) for url in generate_paged_urls(url, total_jobs))
            return CrawlerResult(url, jobs, 'list')

        # If there are available filters, pick the next one in FULL_FILTER_ORDER
        available_filter_names = [f[0] for f in available_filters]
        next_filter = next((f for f in FULL_FILTER_ORDER if f in available_filter_names), None)

        if not next_filter:
            print("No available filters in order, generating job detail tasks...")
            jobs.extend(CrawlerJob(url, linkedin_job_crawler) for url in generate_paged_urls(url, total_jobs))
            return CrawlerResult(url, jobs, 'list')

        # Generate new URLs with the next filter applied
        filtered_urls = extend_url_with_filter(url, next_filter)
        # Go through each new URL and create a CrawlerJob for it
        for filtered_url in filtered_urls:
            jobs.append(CrawlerJob(filtered_url, linkedin_page_crawler))

        return CrawlerResult(url, jobs, 'list')
    else:
        print("Less than or equal to 1000 jobs, generating job detail tasks...")
        jobs.extend(CrawlerJob(url, linkedin_job_crawler) for url in generate_paged_urls(url, total_jobs))
        return CrawlerResult(url, jobs, 'list')

def linkedin_job_crawler(driver, url, time_sleep=1, wait_time=10) -> CrawlerResult:
    page_data = get_linkedin_job_main_page(driver, url, time_sleep, wait_time, scroll=True)
    if not page_data:
        return CrawlerResult(url, [], 'detail')

    # Wait for the first job card to load
    if not wait_for_element(driver, "ul:first-of-type>li.ember-view div.artdeco-entity-lockup__metadata", timeout=wait_time):
        return CrawlerResult(url, [], 'detail')

    job_cards = page_data.find_elements(By.CSS_SELECTOR, "ul:first-of-type>li.ember-view")
    jobs = [extract_job_data(card) for card in job_cards]

    return CrawlerResult(url, jobs, 'detail')

def linkedin_job_detail_crawler(driver, url, time_sleep=1, wait_time=10):
    pass

