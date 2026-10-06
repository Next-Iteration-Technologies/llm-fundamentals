import archive


def test_search_finds_a_document_with_its_location():
    result = archive.search_document("1234")
    assert result["found"] is True
    assert result["workspace"] == "Finance"
    assert result["link"].endswith("/1234")


def test_search_reports_a_missing_document():
    assert archive.search_document("5678") == {"found": False, "doc_id": "5678"}


def test_access_request_goes_to_the_workspace_owner():
    request = archive.raise_access_request("7001", "Need it for an audit", "priya")
    assert request["created"] is True
    assert request["approver"] == "hr-records-lead"
    assert request["number"].startswith("RITM")


def test_upload_status_accepts_a_first_name():
    uploads = archive.check_upload_status(uploaded_by="Priya Sharma")["uploads"]
    assert uploads[0]["archived"] == 18
    assert len(uploads[0]["failed"]) == 2


def test_retention_info_has_a_deletion_date():
    assert archive.get_retention_info("1234")["delete_after"] == "2035-12-31"
