import requests
from typing import Dict, List
import os
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PRReviewComment:
    user: str
    body: str
    created_at: datetime
    path: str
    position: int

@dataclass
class PRReview:
    user: str
    state: str
    body: str
    comments: List[PRReviewComment]
    submitted_at: datetime

@dataclass
class PullRequest:
    title: str
    body: str
    state: str
    created_at: datetime
    updated_at: datetime
    reviews: List[PRReview]
    base_branch: str
    head_branch: str

class GitHubPRReader:
    def __init__(self, token: str):
        """Initialize with GitHub personal access token"""
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'

    def parse_pr_url(self, pr_url: str) -> tuple:
        """Extract owner, repo and PR number from GitHub PR URL"""
        parts = pr_url.strip('/').split('/')
        owner = parts[-4]
        repo = parts[-3]
        pr_number = parts[-1]
        return owner, repo, pr_number

    def get_pr_details(self, pr_url: str) -> PullRequest:
        """Get all PR details including reviews"""
        owner, repo, pr_number = self.parse_pr_url(pr_url)
        
        # Get PR base info
        pr_endpoint = f'{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}'
        pr_response = requests.get(pr_endpoint, headers=self.headers)
        pr_data = pr_response.json()

        # Get PR reviews
        reviews_endpoint = f'{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/reviews'
        reviews_response = requests.get(reviews_endpoint, headers=self.headers)
        reviews_data = reviews_response.json()

        # Process reviews and their comments
        reviews = []
        for review in reviews_data:
            # Get review comments
            comments_endpoint = f'{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/reviews/{review["id"]}/comments'
            comments_response = requests.get(comments_endpoint, headers=self.headers)
            comments_data = comments_response.json()

            review_comments = [
                PRReviewComment(
                    user=comment['user']['login'],
                    body=comment['body'],
                    created_at=datetime.fromisoformat(comment['created_at'].replace('Z', '+00:00')),
                    path=comment['path'],
                    position=comment.get('position', 0)
                )
                for comment in comments_data
            ]

            reviews.append(PRReview(
                user=review['user']['login'],
                state=review['state'],
                body=review.get('body', ''),
                comments=review_comments,
                submitted_at=datetime.fromisoformat(review['submitted_at'].replace('Z', '+00:00'))
            ))

        return PullRequest(
            title=pr_data['title'],
            body=pr_data['body'] or '',
            state=pr_data['state'],
            created_at=datetime.fromisoformat(pr_data['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(pr_data['updated_at'].replace('Z', '+00:00')),
            reviews=reviews,
            base_branch=pr_data['base']['ref'],
            head_branch=pr_data['head']['ref']
        )

def main():
    # Get GitHub token from environment variable
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        raise ValueError("Please set GITHUB_TOKEN environment variable")

    # Initialize reader
    reader = GitHubPRReader(github_token)

    # Example PR URL
    pr_url = input("Enter GitHub PR URL: ")
    
    try:
        pr_details = reader.get_pr_details(pr_url)
        
        # Print structured output
        print(f"\nPull Request: {pr_details.title}")
        print(f"State: {pr_details.state}")
        print(f"Created: {pr_details.created_at}")
        print(f"Updated: {pr_details.updated_at}")
        print(f"Base Branch: {pr_details.base_branch}")
        print(f"Head Branch: {pr_details.head_branch}")
        print("\nDescription:")
        print(pr_details.body)
        
        print("\nReviews:")
        for review in pr_details.reviews:
            print(f"\n{review.user} - {review.state} - {review.submitted_at}")
            if review.body:
                print(f"Review comment: {review.body}")
            
            if review.comments:
                print("\nInline comments:")
                for comment in review.comments:
                    print(f"- File: {comment.path}, Position: {comment.position}")
                    print(f"  {comment.user}: {comment.body}")
                    print(f"  Posted at: {comment.created_at}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 