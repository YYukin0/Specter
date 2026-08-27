"""
Small hermetic summarization set for the P1.3 downstream-task parity check.

These passages and reference summaries are written from scratch for this project
(neutral encyclopedic topics) so the parity run needs no dataset download and
raises no copyright question. They are deliberately short so a 1.5B target model
can produce a reasonable one-sentence summary and so the whole parity sweep runs
in a few minutes on a 24GB Mac.
"""

PARITY_ITEMS = [
    {
        "text": (
            "Photosynthesis is the process by which green plants, algae, and some "
            "bacteria convert light energy into chemical energy. Using sunlight, "
            "water drawn up from the roots, and carbon dioxide taken in through "
            "pores in the leaves, they produce glucose and release oxygen as a "
            "by-product. The reactions take place mainly in the chloroplasts, "
            "which contain the green pigment chlorophyll."
        ),
        "summary": (
            "Photosynthesis lets plants and some microbes turn sunlight, water, "
            "and carbon dioxide into glucose while releasing oxygen."
        ),
    },
    {
        "text": (
            "The water cycle describes how water moves continuously between the "
            "oceans, the atmosphere, and the land. Heat from the sun evaporates "
            "water from seas and lakes; the vapour rises, cools, and condenses "
            "into clouds; and it eventually falls back as rain or snow. Some of "
            "that precipitation soaks into the ground while the rest runs off "
            "through rivers back to the sea."
        ),
        "summary": (
            "The water cycle is the sun-driven movement of water between ocean, "
            "air, and land through evaporation, condensation, and precipitation."
        ),
    },
    {
        "text": (
            "Plate tectonics is the theory that the Earth's outer shell is "
            "divided into large slabs called plates that glide slowly over the "
            "mantle. Where plates pull apart, new crust forms; where they collide, "
            "one plate may slide beneath another and build mountains; and where "
            "they slip past each other, earthquakes are common. The idea unified "
            "earlier observations about drifting continents and sea-floor spreading."
        ),
        "summary": (
            "Plate tectonics explains earthquakes, mountains, and new crust as "
            "results of rigid plates moving over the mantle."
        ),
    },
    {
        "text": (
            "Vaccines work by training the immune system to recognise a pathogen "
            "before a real infection occurs. A vaccine contains a harmless piece "
            "or weakened form of the microbe, which prompts the body to build "
            "antibodies and memory cells. If the person later meets the real "
            "pathogen, their immune system responds quickly enough to prevent or "
            "lessen the disease."
        ),
        "summary": (
            "Vaccines expose the immune system to a harmless version of a pathogen "
            "so it can respond fast to a later real infection."
        ),
    },
    {
        "text": (
            "The printing press developed by Johannes Gutenberg in the fifteenth "
            "century used movable metal type that could be rearranged and reused. "
            "It made books far cheaper and faster to produce than hand copying, "
            "which helped spread literacy, standardise texts, and accelerate the "
            "exchange of ideas across Europe."
        ),
        "summary": (
            "Gutenberg's movable-type press made books cheap to produce and "
            "helped spread literacy and ideas across Europe."
        ),
    },
    {
        "text": (
            "A black hole is a region of space where gravity is so strong that "
            "nothing, not even light, can escape once it crosses the boundary "
            "called the event horizon. Black holes form when a massive star "
            "collapses at the end of its life. Although the hole itself emits no "
            "light, astronomers detect it by the radiation from gas heated as it "
            "spirals inward and by its gravitational pull on nearby stars."
        ),
        "summary": (
            "A black hole is a collapsed massive star whose gravity traps light, "
            "detected by its pull and by glowing infalling gas."
        ),
    },
    {
        "text": (
            "Antibiotic resistance arises when bacteria evolve mechanisms to "
            "survive drugs that once killed them. Overuse and misuse of "
            "antibiotics in medicine and agriculture speed up this process by "
            "killing susceptible bacteria and leaving resistant ones to multiply. "
            "The result is infections that are harder and more expensive to treat."
        ),
        "summary": (
            "Overusing antibiotics drives bacteria to evolve resistance, making "
            "infections harder to treat."
        ),
    },
    {
        "text": (
            "The Great Barrier Reef, off the coast of Queensland, Australia, is "
            "the world's largest coral reef system, stretching more than two "
            "thousand kilometres. It is built by billions of tiny animals called "
            "coral polyps and supports thousands of species of fish, molluscs, "
            "and marine mammals. Rising sea temperatures have caused repeated "
            "mass bleaching events that threaten the reef's survival."
        ),
        "summary": (
            "The Great Barrier Reef is the world's largest coral system, rich in "
            "marine life but increasingly harmed by warming-driven bleaching."
        ),
    },
    {
        "text": (
            "Compound interest is interest calculated on both the original "
            "principal and the interest that has already accumulated. Because each "
            "period's interest is added to the base for the next period, savings "
            "or debts grow faster over time than they would under simple interest. "
            "The effect becomes more pronounced the longer the money is left to "
            "compound."
        ),
        "summary": (
            "Compound interest is paid on principal plus past interest, so "
            "balances grow faster the longer they compound."
        ),
    },
    {
        "text": (
            "The internet is a global network of interconnected computer networks "
            "that communicate using a common set of protocols. Data is broken into "
            "small packets, routed independently across many possible paths, and "
            "reassembled at the destination. This packet-switched design lets the "
            "network keep working even when individual links or routers fail."
        ),
        "summary": (
            "The internet routes data as independently forwarded packets across "
            "interconnected networks, staying resilient when parts fail."
        ),
    },
]
