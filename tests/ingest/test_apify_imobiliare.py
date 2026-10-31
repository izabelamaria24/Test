import pytest

from realestate.ingest.apify_imobiliare import fetch_imobiliare_listings, sector_search_url


class _Response:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self) -> None:
        pass

    def json(self):
        return self._body


class FakeApify:
    def __init__(self, statuses: list[str], items: list[dict]):
        self._statuses = statuses
        self._items = items
        self.calls: list[tuple[str, str, dict]] = []

    def post(self, url, *, params, json, headers, timeout):
        self.calls.append(("POST", url, {"params": params, "json": json, "headers": headers}))
        return _Response({"data": {"id": "RUN1", "status": "READY", "defaultDatasetId": "DS1"}})

    def get(self, url, *, params, headers, timeout):
        self.calls.append(("GET", url, {"params": params, "headers": headers}))
        if url.endswith("/actor-runs/RUN1"):
            status = self._statuses.pop(0)
            run = {"id": "RUN1", "status": status, "defaultDatasetId": "DS1"}
            if status == "SUCCEEDED":
                run["usageTotalUsd"] = 0.0142
            return _Response({"data": run})
        return _Response(self._items)


def _fetch(apify: FakeApify, **kwargs):
    return fetch_imobiliare_listings(
        "secret-token",
        offer_type=kwargs.pop("offer_type", "rent"),
        max_items=kwargs.pop("max_items", 20),
        max_charge_usd=kwargs.pop("max_charge_usd", 0.5),
        post_fn=apify.post,
        get_fn=apify.get,
        **kwargs,
    )


def test_a_search_url_replaces_the_default_city_search():
    apify = FakeApify(["SUCCEEDED"], items=[])
    url = "https://www.imobiliare.ro/inchirieri-apartamente/bucuresti/sector-3"

    _fetch(apify, start_url=url)

    assert apify.calls[0][2]["json"]["startUrls"] == [{"url": url}]


@pytest.mark.parametrize(
    ("offer_type", "sector", "expected"),
    [
        ("sell", 1, "https://www.imobiliare.ro/vanzare-apartamente/bucuresti/sector-1"),
        ("rent", 6, "https://www.imobiliare.ro/inchirieri-apartamente/bucuresti/sector-6"),
    ],
)
def test_sector_search_url(offer_type, sector, expected):
    assert sector_search_url(offer_type, sector) == expected


@pytest.mark.parametrize(("offer_type", "sector"), [("sell", 0), ("sell", 7), ("buy", 2)])
def test_sector_search_url_rejects_bad_input(offer_type, sector):
    with pytest.raises(ValueError):
        sector_search_url(offer_type, sector)


def test_starts_a_capped_run_and_returns_its_listings():
    apify = FakeApify(["RUNNING", "SUCCEEDED"], items=[{"id": "1"}, {"id": "2"}])

    result = _fetch(apify)

    assert result.listings == [{"id": "1"}, {"id": "2"}]
    assert result.cost_usd == 0.0142
    method, url, start = apify.calls[0]
    assert method == "POST"
    assert url == "https://api.apify.com/v2/acts/unfenced-group~imobiliare-ro-scraper/runs"
    assert start["json"] == {
        "offerType": "rent",
        "propertyType": "apartamente",
        "location": "bucuresti",
        "maxItems": 20,
        "fetchDetails": True,
    }
    assert start["params"]["maxTotalChargeUsd"] == 0.5
    assert start["params"]["maxItems"] == 20
    assert start["headers"] == {"Authorization": "Bearer secret-token"}
    assert apify.calls[-1][1] == "https://api.apify.com/v2/datasets/DS1/items"


def test_waits_on_the_server_instead_of_polling_quickly():
    apify = FakeApify(["RUNNING", "RUNNING", "SUCCEEDED"], items=[])

    _fetch(apify)

    waits = [call[2]["params"] for call in apify.calls if call[1].endswith("/actor-runs/RUN1")]
    assert waits == [{"waitForFinish": 60}] * 3


@pytest.mark.parametrize("status", ["FAILED", "TIMED-OUT", "ABORTED"])
def test_raises_when_the_run_does_not_succeed(status):
    with pytest.raises(RuntimeError, match=status):
        _fetch(FakeApify([status], items=[]))


@pytest.mark.parametrize("offer_type", ["buy", ""])
def test_rejects_unknown_offer_types(offer_type):
    with pytest.raises(ValueError):
        _fetch(FakeApify([], items=[]), offer_type=offer_type)
