import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # raise NotImplementedError
    model = {}
    links = corpus.get(page, [])
    if len(links) == 0:
        for p in corpus:
            model[p] = 1 / len(corpus)
    else:
        for p in corpus:
            model[p] = (1 - damping_factor) / len(corpus)
            if p in links:
                model[p] += damping_factor * (1 / len (links))
    return model

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    if corpus is None:
        raise ValueError("corpus is empty")
    samples = []
    transition_models = {}
    for page in corpus:
        transition_models[page] = transition_model(corpus, page, damping_factor)
    # generate the first sample
    samples.append(random.choice(list(corpus.keys())))
    # generate the rest of the samples
    for i in range(1, n):
        transition_ = transition_models.get(samples[i-1], None)
        samples.append(random.choices(population=list(transition_.keys()), weights=list(transition_.values()), k=1)[0])
    
    page_rank = {}
    for page in samples:
        page_rank[page] = page_rank.get(page, 0) + 1
    for page in page_rank:
        page_rank[page] /= n
    return page_rank

def is_converged(previous_page_rank, current_page_rank):
    for page in previous_page_rank:
        if current_page_rank[page] - previous_page_rank[page] > 0.001:
            return False
    return True

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    if corpus is None:
        raise ValueError("corpus is empty")
    incoming_pages = {}
    # get all pages with empty links
    non_link_pages = set(page for page in corpus if len(corpus[page]) == 0)
    for page in corpus:
        incoming_pages[page] = non_link_pages.copy()

    for page in corpus:
        links = corpus.get(page, [])
        for link in links:
            incoming_pages[link] = incoming_pages.get(link, set())
            incoming_pages[link].add(page)
    previous_page_rank = {page: -1 for page in corpus}
    current_page_rank = {page: 1/len(corpus) for page in corpus}
    while not is_converged(previous_page_rank, current_page_rank):
        previous_page_rank = current_page_rank.copy()
        for page in incoming_pages:
            current_page_rank[page] =  (1 - damping_factor) / len(corpus)
            for p in incoming_pages[page]:
                if len(corpus[p]) == 0:
                    current_page_rank[page] += damping_factor * (previous_page_rank[p] / len(corpus))
                else:
                    current_page_rank[page] += damping_factor * (previous_page_rank[p] / len(corpus[p]))
    return current_page_rank

if __name__ == "__main__":
    main()