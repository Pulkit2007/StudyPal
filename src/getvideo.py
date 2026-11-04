from youtubesearchpython import VideosSearch


def getvideo(query: str):

    video_search = VideosSearch(query=query, limit=3)
    result = video_search.result()
    video_title = [video["title"] for video in result["result"]]
    video_links = [video["link"] for video in result["result"]]
    return video_title, video_links


user_query = "linear algebra"
video_titles, video_links = getvideo(user_query)
print(video_titles, video_links)
