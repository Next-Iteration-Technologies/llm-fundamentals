"""Sales-tax placeholder used in workshop examples."""

def tax_rate_for_country(country: str) -> float:
    """Return a demo tax rate. Unknown countries return 0.0."""
    rates = {"IN": 0.18, "US": 0.0, "DE": 0.19}
    return rates.get(country.upper(), 0.0)
