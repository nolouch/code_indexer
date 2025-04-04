import requests
from typing import Dict, List
import os
from dataclasses import dataclass
from datetime import datetime
import base64
import re


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

    def format(self, format_type: str = "text") -> str:
        """
        Format the pull request data into different representations.

        Args:
            format_type: The type of formatting to use ("text", "markdown", "json")

        Returns:
            A formatted string representation of the pull request
        """
        if format_type == "markdown":
            # Markdown format
            output = [
                f"# {self.title}",
                f"**State:** {self.state}  ",
                f"**Created:** {self.created_at}  ",
                f"**Updated:** {self.updated_at}  ",
                f"**Base Branch:** {self.base_branch}  ",
                f"**Head Branch:** {self.head_branch}  ",
                "\n## Description",
                self.body,
                "\n## Reviews",
            ]

            for review in self.reviews:
                output.append(f"\n### {review.user} - {review.state}")
                output.append(f"*Submitted at: {review.submitted_at}*")
                if review.body:
                    output.append(f"\n{review.body}")

                if review.comments:
                    output.append("\n#### Inline Comments")
                    for comment in review.comments:
                        output.extend(
                            [
                                f"- **{comment.path}** (Position: {comment.position})",
                                f"  > {comment.user}: {comment.body}",
                                f"  > *Posted at: {comment.created_at}*",
                            ]
                        )

            return "\n".join(output)

        else:
            # Basic text format
            output = [
                f"Pull Request: {self.title}",
                f"State: {self.state}",
                f"Created: {self.created_at}",
                f"Updated: {self.updated_at}",
                f"Base Branch: {self.base_branch}",
                f"Head Branch: {self.head_branch}",
                "\nDescription:",
                self.body,
                "\nReviews:",
            ]

            for review in self.reviews:
                output.append(
                    f"\n{review.user} - {review.state} - {review.submitted_at}"
                )
                if review.body:
                    output.append(f"Review comment: {review.body}")

                if review.comments:
                    output.append("\nInline comments:")
                    for comment in review.comments:
                        output.extend(
                            [
                                f"- File: {comment.path}, Position: {comment.position}",
                                f"  {comment.user}: {comment.body}",
                                f"  Posted at: {comment.created_at}",
                            ]
                        )

            return "\n".join(output)


class GitHubPRReader:
    def __init__(self, token: str):
        """Initialize with GitHub personal access token"""
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
        self.base_url = "https://api.github.com"

    def parse_pr_url(self, pr_url: str) -> tuple:
        """Extract owner, repo and PR number from GitHub PR URL"""
        parts = pr_url.strip("/").split("/")
        owner = parts[-4]
        repo = parts[-3]
        pr_number = parts[-1]
        return owner, repo, pr_number

    def get_pr_details(self, pr_url: str) -> PullRequest:
        """Get all PR details including reviews"""
        owner, repo, pr_number = self.parse_pr_url(pr_url)

        # Get PR base info
        pr_endpoint = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}"
        pr_response = requests.get(pr_endpoint, headers=self.headers)
        pr_data = pr_response.json()

        # Get PR reviews
        reviews_endpoint = (
            f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
        )
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
                    user=comment["user"]["login"],
                    body=comment["body"],
                    created_at=datetime.fromisoformat(
                        comment["created_at"].replace("Z", "+00:00")
                    ),
                    path=comment["path"],
                    position=comment.get("position", 0),
                )
                for comment in comments_data
            ]

            reviews.append(
                PRReview(
                    user=review["user"]["login"],
                    state=review["state"],
                    body=review.get("body", ""),
                    comments=review_comments,
                    submitted_at=datetime.fromisoformat(
                        review["submitted_at"].replace("Z", "+00:00")
                    ),
                )
            )

        return PullRequest(
            title=pr_data["title"],
            body=pr_data["body"] or "",
            state=pr_data["state"],
            created_at=datetime.fromisoformat(
                pr_data["created_at"].replace("Z", "+00:00")
            ),
            updated_at=datetime.fromisoformat(
                pr_data["updated_at"].replace("Z", "+00:00")
            ),
            reviews=reviews,
            base_branch=pr_data["base"]["ref"],
            head_branch=pr_data["head"]["ref"],
        )

    def read_github_file(self, github_url):
        """
        Read content from a GitHub file URL

        Args:
            github_url: URL to a GitHub file (like https://github.com/owner/repo/blob/branch/path/to/file)

        Returns:
            The content of the file as string
        """
        # Parse the GitHub URL to extract owner, repo, branch and path
        pattern = r"https://github.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)"
        match = re.match(pattern, github_url)

        if not match:
            raise ValueError(f"Invalid GitHub URL format: {github_url}")

        owner, repo, branch, path = match.groups()

        # Construct the GitHub API URL
        api_url = (
            f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
        )

        # Set up headers with GitHub token if available
        headers = {}
        import os

        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        # Make the API request
        response = requests.get(api_url, headers=headers)

        if response.status_code != 200:
            raise Exception(
                f"Failed to fetch file: {response.status_code} - {response.text}"
            )

        # Parse the response and decode content
        data = response.json()
        content = base64.b64decode(data["content"]).decode("utf-8")

        return content
