import requests
import time


def test_api():
    base_url = "http://localhost:8051"

    response = requests.get(f"{base_url}/")
    assert response.status_code == 200

    query = {"query": "test query"}
    response = requests.post(f"{base_url}/search", json=query)
    assert response.status_code == 200

    data = response.json()
    assert "rel_docs" in data
    assert "rel_docs_sim" in data


if __name__ == "__main__":
    time.sleep(5)
    test_api()
    print("All tests passed!")
