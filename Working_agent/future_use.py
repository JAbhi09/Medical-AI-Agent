# We can use this advance function in medical_ner_piepline for future use to look up for the term.

def _search_umls_concept_enhanced(self, term: str) -> List[Dict]:
    service_ticket = self._get_fresh_service_ticket()
    if not service_ticket:
        return []

    search_strategies = [
        # Strategy 1: Exact match, any vocabulary
        {
            'ticket': service_ticket,
            'string': term,
            'searchType': 'exact',
            'returnIdType': 'sourceUi',
            'pageSize': 10
        },
        # Strategy 2: Approximate match, any vocabulary
        {
            'ticket': None,  # Will get fresh ticket
            'string': term,
            'searchType': 'approximate',
            'returnIdType': 'sourceUi',
            'pageSize': 10
        },
        # Strategy 3: Word search with SNOMED CT
        {
            'ticket': None,  # Will get fresh ticket
            'string': term,
            'searchType': 'words',
            'sabs': 'SNOMEDCT_US',
            'returnIdType': 'sourceUi',
            'pageSize': 10
        },
        # Strategy 4: Word search with MeSH
        {
            'ticket': None,  # Will get fresh ticket
            'string': term,
            'searchType': 'words',
            'sabs': 'MSH',
            'returnIdType': 'sourceUi',
            'pageSize': 10
        }
    ]

    for i, params in enumerate(search_strategies):
        try:
            # Get fresh ticket for each strategy (except first)
            if params['ticket'] is None:
                fresh_ticket = self._get_fresh_service_ticket()
                if not fresh_ticket:
                    continue
                params['ticket'] = fresh_ticket

            logger.debug(f"Trying search strategy {i + 1} for '{term}'")
            response = requests.get(self.search_endpoint, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()
                results = data.get('result', {}).get('results', [])

                if results:
                    logger.debug(f"Strategy {i + 1} found {len(results)} results for '{term}'")
                    return results
            else:
                logger.debug(f"Strategy {i + 1} failed: {response.status_code}")

        except Exception as e:
            logger.debug(f"Strategy {i + 1} error: {e}")
            continue

    return []