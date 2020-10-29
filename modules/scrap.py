import newspaper  # from newspaper import Article, Config # https://newspaper.readthedocs.io/en/latest/   
from datetime import datetime
from urllib.parse import urljoin, urlparse, parse_qs
import re
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 콘솔로 출력
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def scraper(urls, idx=None):    
    docs_info = []
    docs_idx = []
    error_urls_by_types = {'parse_error':[], 'empty_contents':[], 'none_contents':[]}

    
    if idx is not None:
        assert len(idx) == len(urls), "The length of urls and idx should be same."
    doc_info = {}
    #urls = set(urls)  # 이거 실행하면 순서가 바뀌어버림..
    
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    config = newspaper.Config()
    config.browser_user_agent = user_agent

    for i, url in tqdm(enumerate(urls), desc='scraper'):
        url = url_prefix_adder(url)
        article = newspaper.Article(url, config=config)  # , language='ko'

        try:
            logger.info("loading %s", url)
            article.download()  # request
            article.parse()  # parsing
            
            doc_info = {
                'title': article.title,
    #             'authors': article.authors,
                'publish_date': article.publish_date,
                'contents': article.text,
                'url': url,
                'scrap_at': datetime.now(),
                'is_news': article.is_valid_url(),
    #             'top_image': article.top_image,
    #             'movies': article.movies
            }

        except:
            logger.error(f"Cannot parse, {url}")
            error_urls_by_types['parse_error'].append(url)
            continue
            
        if doc_info['title'] == '' or doc_info['contents'] == '':
            logger.error("title/contents is empty, %s", str(url))
            error_urls_by_types['empty_contents'].append(url)
            continue
        elif doc_info['title'] == None or doc_info['contents'] == None:
            logger.error("title/contents is None, %s", str(url))
            error_urls_by_types['none_contents'].append(url)
            continue
        else:
            #print(doc_info['title'], type(doc_info['title']))
            docs_info.append(doc_info)      
            if idx is not None:
                docs_idx.append(idx[i])

    logger.info(f"Complete scrape {len(docs_info)} among {len(urls)}")
        
    return docs_info, docs_idx, error_urls_by_types


def url_extractor(text):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,text)       
    return [x[0] for x in url] 

def url_prefix_adder(url):
    """
    url 앞에 http:// 또는 https://가 붙어 있지 않은 url의 경우, 앞에 이를 붙여서 리턴
    """
    http_reg = re.compile("https?://\S*")
    
    if http_reg.match(url):
        return url
    else:
        url_fixed = "https://" + url  #http로..?
        # requests.get() # 잘되는지 체크가 필요하지만...    
        return url_fixed
