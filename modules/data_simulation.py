"""
Synthetic product catalog generator for RetailMind.

Generates a curated catalog of ~200 realistic e-commerce products with rich
descriptions, material specs, star ratings, and semantic tags — designed to
produce high-quality embeddings for dense retrieval.
"""

import random
from typing import TypedDict

random.seed(42)  # Reproducible catalog across sessions


class Product(TypedDict):
    id: int
    title: str
    category: str
    price: float
    desc: str
    tags: list[str]
    rating: float
    reviews: int
    materials: str


# ---------------------------------------------------------------------------
# Hand-authored product templates — each with unique, embedding-rich content
# ---------------------------------------------------------------------------

_TEMPLATES: list[dict] = [
    # ── Winter ──────────────────────────────────────────────────────────────
    {"title": "Alpine Pro Insulated Parka", "category": "winter", "price": 189.99,
     "desc": "Engineered for sub-zero temperatures with 700-fill goose down insulation and a waterproof shell. Features an adjustable storm hood, internal media pocket, and reflective accents for low-light visibility. Wind-rated to -30°F.",
     "tags": ["waterproof", "insulated", "cold-weather", "outdoor"], "materials": "Nylon ripstop shell, goose down fill"},
    {"title": "Fireside Merino Wool Sweater", "category": "winter", "price": 79.99,
     "desc": "A classic crewneck knit from ultra-soft 100% merino wool. Breathable yet warm, perfect for layering or wearing solo by the fire. Naturally odor-resistant and temperature-regulating.",
     "tags": ["wool", "layering", "classic", "cozy"], "materials": "100% Merino wool"},
    {"title": "Glacier Grip Thermal Gloves", "category": "winter", "price": 34.99,
     "desc": "Touchscreen-compatible thermal gloves with silicone grip palms. Fleece-lined interior keeps hands warm while conductive fingertips let you use your phone without exposing skin to the cold.",
     "tags": ["touchscreen", "thermal", "cold-weather", "tech-friendly"], "materials": "Polyester fleece, silicone grip, conductive thread"},
    {"title": "Blizzard Shield Snow Boots", "category": "winter", "price": 149.99,
     "desc": "Heavy-duty winter boots with Thinsulate insulation and Vibram Arctic Grip outsoles. Sealed seams and a gusseted tongue keep snow and slush out. Comfort-rated to -40°F.",
     "tags": ["waterproof", "insulated", "snow", "hiking"], "materials": "Full-grain leather, Thinsulate, Vibram sole"},
    {"title": "Nordic Knit Beanie", "category": "winter", "price": 24.99,
     "desc": "Double-layer acrylic knit beanie with a fleece headband liner. Classic Nordic pattern adds style while the snug fit traps heat. One size fits most.",
     "tags": ["knit", "warm", "casual", "unisex"], "materials": "Acrylic knit, polyester fleece liner"},
    {"title": "Summit Fleece Pullover", "category": "winter", "price": 64.99,
     "desc": "Mid-weight microfleece pullover ideal for layering under a shell or wearing on cool autumn mornings. Quarter-zip design, chin guard, and zippered chest pocket.",
     "tags": ["fleece", "layering", "outdoor", "mid-weight"], "materials": "100% recycled polyester microfleece"},
    {"title": "Thermal Base Layer Set", "category": "winter", "price": 54.99,
     "desc": "Moisture-wicking thermal top and leggings designed as a first layer for skiing, snowboarding, or cold commutes. Flatlock seams prevent chafing during all-day wear.",
     "tags": ["base-layer", "moisture-wicking", "skiing", "thermal"], "materials": "Merino-synthetic blend"},
    {"title": "Expedition Down Vest", "category": "winter", "price": 109.99,
     "desc": "Packable 650-fill down vest that compresses into its own pocket. Provides core warmth without restricting arm movement — perfect for active winter pursuits or travel.",
     "tags": ["packable", "down", "layering", "travel"], "materials": "Water-resistant nylon, 650-fill duck down"},

    # ── Summer ──────────────────────────────────────────────────────────────
    {"title": "Breeze Runner Mesh Sneakers", "category": "summer", "price": 89.99,
     "desc": "Ultra-breathable mesh upper with a responsive foam midsole. Weighs just 7.2 oz per shoe, making them ideal for hot-weather runs, gym sessions, or all-day wear in the heat.",
     "tags": ["breathable", "lightweight", "running", "mesh"], "materials": "Engineered mesh upper, EVA foam midsole"},
    {"title": "Pacific Coast Board Shorts", "category": "summer", "price": 39.99,
     "desc": "Quick-dry board shorts with a 4-way stretch waistband and secure zip pocket. UPF 50+ sun protection fabric keeps you safe from UV rays during long beach days.",
     "tags": ["quick-dry", "UPF", "beach", "swim"], "materials": "Recycled polyester, elastane blend"},
    {"title": "Solaris UV Shield Sunglasses", "category": "summer", "price": 59.99,
     "desc": "Polarized lenses with 100% UV400 protection in a lightweight titanium frame. Anti-glare coating reduces eye strain on bright days. Comes with a hard-shell carrying case.",
     "tags": ["polarized", "UV-protection", "lightweight", "outdoor"], "materials": "Titanium frame, polarized polycarbonate lenses"},
    {"title": "Coastal Breeze Linen Shirt", "category": "summer", "price": 49.99,
     "desc": "Relaxed-fit linen button-down that stays cool in 90°F+ heat. Garment-dyed for a lived-in look. Perfect from boardwalk brunch to sunset cocktails.",
     "tags": ["linen", "breathable", "casual", "warm-weather"], "materials": "100% French linen"},
    {"title": "Reef Walker Sandals", "category": "summer", "price": 44.99,
     "desc": "Contoured footbed sandals with arch support and a rugged outsole. Synthetic nubuck straps adjust for a custom fit. Great for beach walks, pool decks, and casual summer outings.",
     "tags": ["sandals", "arch-support", "beach", "casual"], "materials": "Synthetic nubuck, molded EVA footbed"},
    {"title": "Tropic Mesh Tank Top", "category": "summer", "price": 22.99,
     "desc": "Lightweight mesh-back tank with moisture-wicking fabric that keeps you dry during hot workouts or humid commutes. Flatlock seams and a relaxed hem for all-day comfort.",
     "tags": ["moisture-wicking", "gym", "breathable", "lightweight"], "materials": "Polyester-spandex blend"},
    {"title": "Sun Shield Wide Brim Hat", "category": "summer", "price": 34.99,
     "desc": "UPF 50+ wide-brim sun hat with an adjustable chin cord and mesh ventilation panels. Floats in water and packs flat for travel. Essential protection for hiking, fishing, and gardening.",
     "tags": ["UPF", "sun-protection", "outdoor", "packable"], "materials": "Nylon with mesh vents"},
    {"title": "Aqua Sport Water Shoes", "category": "summer", "price": 29.99,
     "desc": "Drainage-port water shoes with a grippy rubber sole for rocky beaches and river crossings. Neoprene collar prevents sand entry. Dries in under an hour.",
     "tags": ["water-shoes", "quick-dry", "beach", "outdoor"], "materials": "Mesh, neoprene, rubber outsole"},

    # ── Eco-Friendly ────────────────────────────────────────────────────────
    {"title": "EcoLoop Recycled Backpack", "category": "eco-friendly", "price": 74.99,
     "desc": "Made from 20 recycled ocean-bound plastic bottles. Features a padded laptop sleeve, water-resistant coating, and ergonomic shoulder straps. Every purchase funds 1 lb of ocean cleanup.",
     "tags": ["recycled", "ocean-plastic", "sustainable", "laptop"], "materials": "Recycled RPET fabric, plant-based waterproof coating"},
    {"title": "Bamboo Hydration Bottle", "category": "eco-friendly", "price": 28.99,
     "desc": "Double-wall vacuum insulated bottle with a natural bamboo cap and silicone seal. Keeps drinks cold for 24 hours or hot for 12. BPA-free, plastic-free, and designed to last a lifetime.",
     "tags": ["bamboo", "BPA-free", "insulated", "reusable"], "materials": "18/8 stainless steel, bamboo lid"},
    {"title": "Organic Cotton Classic Tee", "category": "eco-friendly", "price": 32.99,
     "desc": "GOTS-certified organic cotton tee dyed with low-impact, water-saving dyes. Pre-shrunk ring-spun cotton feels buttery soft from the first wear. Fair Trade certified production.",
     "tags": ["organic", "fair-trade", "GOTS-certified", "cotton"], "materials": "100% GOTS organic cotton"},
    {"title": "Hemp Canvas Tote Bag", "category": "eco-friendly", "price": 19.99,
     "desc": "Durable hemp canvas tote that replaces 700 single-use plastic bags in its lifetime. Reinforced seams, interior pocket, and long handles for comfortable shoulder carry.",
     "tags": ["hemp", "reusable", "sustainable", "zero-waste"], "materials": "Organic hemp canvas"},
    {"title": "Plant-Based Running Shoes", "category": "eco-friendly", "price": 119.99,
     "desc": "The upper is woven from eucalyptus fiber, the midsole from sugarcane-based EVA, and the outsole from natural rubber. Carbon-negative manufacturing. Feels like running on clouds.",
     "tags": ["plant-based", "carbon-negative", "running", "vegan"], "materials": "Eucalyptus fiber, sugarcane EVA, natural rubber"},
    {"title": "Recycled Denim Jacket", "category": "eco-friendly", "price": 89.99,
     "desc": "Classic trucker jacket made from 100% post-consumer recycled denim. Each jacket diverts 1.5 lbs of textile waste from landfills. Stone-washed finish with brass buttons.",
     "tags": ["recycled", "denim", "upcycled", "sustainable"], "materials": "100% recycled post-consumer denim"},
    {"title": "Solar-Powered Watch", "category": "eco-friendly", "price": 159.99,
     "desc": "Never needs a battery — charges via any light source. Sapphire crystal face, titanium case, and a strap made from recycled ocean plastic. Water-resistant to 100 meters.",
     "tags": ["solar", "recycled", "titanium", "water-resistant"], "materials": "Titanium, sapphire crystal, recycled ocean-plastic strap"},
    {"title": "Cork Yoga Mat", "category": "eco-friendly", "price": 64.99,
     "desc": "Harvested from sustainable cork oak forests without harming the tree. Non-slip surface improves grip when wet. Antimicrobial naturally. Backed with natural rubber for cushioning.",
     "tags": ["cork", "sustainable", "yoga", "non-toxic"], "materials": "Natural cork, natural rubber backing"},

    # ── Sports & Fitness ────────────────────────────────────────────────────
    {"title": "ProPulse Running Shoes", "category": "sports", "price": 129.99,
     "desc": "Carbon-plate racing shoes with a nitrogen-infused midsole for maximum energy return. Engineered mesh upper weighs just 6.5 oz. Designed for 5K to marathon distances.",
     "tags": ["carbon-plate", "racing", "lightweight", "marathon"], "materials": "Engineered mesh, carbon fiber plate, nitrogen foam"},
    {"title": "FlexCore Training Shorts", "category": "sports", "price": 44.99,
     "desc": "4-way stretch training shorts with a built-in compression liner and three secure pockets. Sweat-wicking DryFit fabric keeps you cool through HIIT, lifting, and sprints.",
     "tags": ["training", "compression", "moisture-wicking", "gym"], "materials": "Polyester-elastane with DryFit technology"},
    {"title": "IronGrip Fitness Watch", "category": "sports", "price": 199.99,
     "desc": "GPS-enabled multisport watch with heart rate monitoring, VO2 max estimation, and 14-day battery life. Tracks 30+ activities including swimming (waterproof to 50m). Syncs with Strava.",
     "tags": ["GPS", "heart-rate", "waterproof", "multisport"], "materials": "Fiber-reinforced polymer case, silicone band"},
    {"title": "Thunder Strike Basketball", "category": "sports", "price": 34.99,
     "desc": "Official size and weight composite leather basketball with deep channel design for superior grip. Indoor/outdoor rated with a butyl bladder for consistent air retention.",
     "tags": ["basketball", "indoor-outdoor", "official-size", "grip"], "materials": "Composite leather, butyl rubber bladder"},
    {"title": "Velocity Compression Tights", "category": "sports", "price": 59.99,
     "desc": "Graduated compression tights that boost blood circulation and reduce muscle fatigue during long runs. Reflective logos for night visibility. Flatlock seams prevent chafing.",
     "tags": ["compression", "running", "reflective", "recovery"], "materials": "Nylon-spandex compression fabric"},
    {"title": "PowerLift Training Gloves", "category": "sports", "price": 27.99,
     "desc": "Ventilated weightlifting gloves with padded leather palms and adjustable wrist wraps. Reduces calluses while maintaining bar feel. Pull-tab for easy removal between sets.",
     "tags": ["weightlifting", "gym", "padded", "grip"], "materials": "Genuine leather palm, mesh back, neoprene wrist wrap"},
    {"title": "AeroFlow Cycling Jersey", "category": "sports", "price": 74.99,
     "desc": "Full-zip cycling jersey with three rear pockets and a silicone gripper hem. Italian mesh side panels maximize airflow on climbs. Sublimation-printed — colors won't fade or peel.",
     "tags": ["cycling", "breathable", "lightweight", "performance"], "materials": "Italian polyester mesh blend"},
    {"title": "Endurance Hydration Pack", "category": "sports", "price": 49.99,
     "desc": "Lightweight 2L hydration vest designed for trail running. Bite valve with on/off switch, front stash pockets for gels, and a bounce-free fit that adjusts with dual sternum straps.",
     "tags": ["hydration", "trail-running", "lightweight", "outdoor"], "materials": "Ripstop nylon, BPA-free reservoir"},

    # ── Electronics & Tech ──────────────────────────────────────────────────
    {"title": "AuraBeats Studio Headphones", "category": "electronics", "price": 249.99,
     "desc": "Active noise cancelling over-ear headphones with 40mm custom drivers and 30-hour battery life. Adaptive EQ auto-tunes to your ear shape. Features multipoint Bluetooth for switching between laptop and phone.",
     "tags": ["ANC", "wireless", "bluetooth", "noise-cancelling"], "materials": "Memory foam cushions, anodized aluminum, protein leather"},
    {"title": "NovaBand Fitness Tracker", "category": "electronics", "price": 49.99,
     "desc": "Slim fitness band with AMOLED display, continuous heart rate monitoring, sleep tracking, and SpO2 sensor. 10-day battery life and swim-proof to 50 meters. Weighs just 22 grams.",
     "tags": ["fitness-tracker", "AMOLED", "heart-rate", "waterproof"], "materials": "Polycarbonate case, silicone band"},
    {"title": "TrueWireless Pro Earbuds", "category": "electronics", "price": 129.99,
     "desc": "In-ear ANC earbuds with transparency mode and spatial audio support. 6-hour playtime per charge, 24 hours total with the wireless charging case. IPX5 sweat-resistant for workouts.",
     "tags": ["ANC", "earbuds", "wireless", "spatial-audio"], "materials": "Medical-grade silicone tips, matte plastic shell"},
    {"title": "Portable Solar Charger Panel", "category": "electronics", "price": 69.99,
     "desc": "Foldable 21W solar panel with dual USB-A and USB-C outputs. Charges a phone in ~2.5 hours of direct sunlight. Carabiner attachment for backpack mounting during hikes.",
     "tags": ["solar", "portable", "USB-C", "outdoor"], "materials": "Monocrystalline silicon, PET laminate, polyester canvas"},
    {"title": "SmartTherm Travel Mug", "category": "electronics", "price": 39.99,
     "desc": "App-connected travel mug with an LED temperature display on the lid. Set your preferred drinking temperature and the mug maintains it for up to 3 hours via battery-powered heating element.",
     "tags": ["smart", "temperature-control", "travel", "app-connected"], "materials": "304 stainless steel, ceramic coating interior"},
    {"title": "UltraSlim Power Bank 10K", "category": "electronics", "price": 34.99,
     "desc": "10,000mAh portable charger thinner than most phones. Dual output (USB-C PD + USB-A QC3.0) charges two devices simultaneously. Fully recharges in 2.5 hours.",
     "tags": ["power-bank", "USB-C", "portable", "fast-charging"], "materials": "Aluminum alloy shell, lithium-polymer cells"},
    {"title": "Compact Bluetooth Speaker", "category": "electronics", "price": 44.99,
     "desc": "IP67 waterproof and dustproof mini speaker with surprisingly rich 360° sound. 12-hour battery, built-in mic for calls, and a carabiner loop. Floats in water.",
     "tags": ["bluetooth", "waterproof", "portable", "speaker"], "materials": "Rubberized exterior, passive bass radiator"},
    {"title": "Night Owl LED Desk Lamp", "category": "electronics", "price": 54.99,
     "desc": "Dimmable LED desk lamp with 5 color temperature presets and a wireless Qi charging pad in the base. Adjustable gooseneck, memory function, and a 1-hour auto-off timer.",
     "tags": ["LED", "desk-lamp", "wireless-charging", "dimmable"], "materials": "Aluminum arm, ABS base with Qi coil"},

    # ── Premium / Luxury ────────────────────────────────────────────────────
    {"title": "Artisan Leather Weekender", "category": "premium", "price": 349.99,
     "desc": "Hand-stitched full-grain vegetable-tanned leather duffle with brass YKK zippers. Develops a rich patina with age. Separate shoe compartment and detachable shoulder strap.",
     "tags": ["leather", "handmade", "luxury", "travel"], "materials": "Full-grain vegetable-tanned leather, brass hardware"},
    {"title": "Heritage Automatic Watch", "category": "premium", "price": 499.99,
     "desc": "Swiss-movement automatic watch with a sapphire crystal dial and exhibition caseback. 42mm stainless steel case with a genuine alligator strap. 50-meter water resistance.",
     "tags": ["automatic", "swiss-movement", "sapphire", "luxury"], "materials": "316L stainless steel, sapphire crystal, alligator leather strap"},
    {"title": "Cashmere Blend Overcoat", "category": "premium", "price": 389.99,
     "desc": "Italian-milled cashmere-wool blend overcoat with a notch lapel and half-canvas construction. Fully lined in Bemberg silk. Timeless silhouette for dressed-up or smart-casual looks.",
     "tags": ["cashmere", "Italian", "luxury", "formal"], "materials": "70% wool, 30% cashmere, Bemberg lining"},
    {"title": "Handcrafted Walnut Sunglasses", "category": "premium", "price": 179.99,
     "desc": "Frames carved from sustainably sourced American black walnut with Carl Zeiss polarized lenses. Each pair has unique wood grain patterns. Spring hinges for a comfortable universal fit.",
     "tags": ["handcrafted", "walnut", "polarized", "sustainable"], "materials": "Black walnut wood, Carl Zeiss polarized lenses"},
    {"title": "Titanium Card Wallet", "category": "premium", "price": 89.99,
     "desc": "Minimalist RFID-blocking wallet machined from grade-5 titanium. Holds 6 cards and features a quick-access pull tab. Weighs just 2.1 oz and will outlast any leather wallet.",
     "tags": ["titanium", "RFID-blocking", "minimalist", "EDC"], "materials": "Grade-5 titanium, Dyneema pull tab"},
    {"title": "Silk Pocket Square Collection", "category": "premium", "price": 59.99,
     "desc": "Set of 3 hand-rolled Italian silk pocket squares in complementary patterns. Each square is individually wrapped in tissue — perfect as a gift or to elevate your suit game.",
     "tags": ["silk", "Italian", "gift", "formal"], "materials": "100% Italian silk, hand-rolled edges"},
    {"title": "Executive Leather Belt", "category": "premium", "price": 119.99,
     "desc": "Single-piece full-grain bridle leather belt with a solid brass buckle. No stitching — the leather is thick enough to hold its shape for decades. Ages beautifully with wear.",
     "tags": ["leather", "brass", "luxury", "classic"], "materials": "Full-grain English bridle leather, solid brass buckle"},
    {"title": "Carbon Fiber Money Clip", "category": "premium", "price": 44.99,
     "desc": "Aerospace-grade carbon fiber money clip with a satin finish. Ultra-lightweight and strong enough to hold 15+ folded bills without losing spring tension over time.",
     "tags": ["carbon-fiber", "minimalist", "EDC", "lightweight"], "materials": "3K twill carbon fiber"},

    # ── Home & Lifestyle ────────────────────────────────────────────────────
    {"title": "Aromatherapy Soy Candle Set", "category": "home", "price": 36.99,
     "desc": "Set of 3 hand-poured soy candles in amber glass jars: Lavender Fields, Cedar & Sage, and Vanilla Bean. 45-hour burn time each. Cotton wicks, no synthetic fragrances.",
     "tags": ["soy", "aromatherapy", "handmade", "non-toxic"], "materials": "100% soy wax, cotton wicks, essential oils"},
    {"title": "Japanese Ceramic Pour-Over Set", "category": "home", "price": 54.99,
     "desc": "Minimalist pour-over coffee dripper with a double-wall ceramic server. The cone's spiral ribs allow optimal coffee bloom. Makes 2-4 cups of clean, nuanced brew.",
     "tags": ["ceramic", "coffee", "Japanese", "minimalist"], "materials": "Hasami porcelain, borosilicate server"},
    {"title": "Weighted Linen Throw Blanket", "category": "home", "price": 79.99,
     "desc": "Stonewashed Belgian linen throw with a comfortable 3 lb weight. Gets softer with every wash. Perfect draped over a sofa or at the foot of the bed. OEKO-TEX certified.",
     "tags": ["linen", "stonewashed", "cozy", "OEKO-TEX"], "materials": "100% Belgian flax linen"},
    {"title": "Walnut & Brass Desk Organizer", "category": "home", "price": 44.99,
     "desc": "Handcrafted desk organizer with solid walnut compartments and brass dividers. Holds pens, cards, phone, and small accessories. Felt-lined base protects desktop surfaces.",
     "tags": ["walnut", "brass", "handcrafted", "office"], "materials": "American black walnut, brushed brass accents"},
    {"title": "Terracotta Herb Planter Trio", "category": "home", "price": 29.99,
     "desc": "Set of 3 terracotta planters with drainage holes and bamboo saucers. Perfect for kitchen windowsill herbs like basil, rosemary, and mint. Hand-finished with a matte glaze.",
     "tags": ["terracotta", "gardening", "kitchen", "handmade"], "materials": "Terracotta clay, bamboo saucers"},
    {"title": "Memory Foam Seat Cushion", "category": "home", "price": 39.99,
     "desc": "Ergonomic U-shaped seat cushion with cooling gel-infused memory foam. Reduces tailbone pressure during long work sessions. Machine-washable velour cover with anti-slip bottom.",
     "tags": ["ergonomic", "memory-foam", "office", "comfort"], "materials": "Gel-infused memory foam, velour cover"},
    {"title": "Minimalist Wall Clock", "category": "home", "price": 49.99,
     "desc": "12-inch silent-sweep wall clock with a birch plywood face and brass hands. No ticking sound — uses a precision quartz movement. Mounts flush with a single nail.",
     "tags": ["minimalist", "silent", "birch", "Scandinavian"], "materials": "Baltic birch plywood, brass hands, quartz movement"},
    {"title": "Turkish Cotton Bath Towel Set", "category": "home", "price": 64.99,
     "desc": "Set of 4 Turkish cotton towels — 2 bath, 2 hand. Long-staple cotton loops absorb 3x their weight in water. Gets fluffier with each wash. OEKO-TEX Standard 100.",
     "tags": ["Turkish-cotton", "absorbent", "OEKO-TEX", "bath"], "materials": "100% long-staple Turkish cotton"},

    # ── Casual / Streetwear ─────────────────────────────────────────────────
    {"title": "Urban Canvas Sneakers", "category": "casual", "price": 59.99,
     "desc": "Classic low-top canvas sneakers with a vulcanized rubber sole for all-day comfort. Metal eyelets, cotton laces, and a removable cushioned insole. Comes in 8 colorways.",
     "tags": ["canvas", "classic", "casual", "street"], "materials": "Organic cotton canvas, vulcanized rubber sole"},
    {"title": "Oversized Graphic Hoodie", "category": "casual", "price": 54.99,
     "desc": "Heavyweight 14 oz French terry hoodie with a relaxed oversized fit. Abstract graphic screen-printed with water-based inks. Ribbed cuffs, kangaroo pocket, and a double-lined hood.",
     "tags": ["hoodie", "oversized", "streetwear", "graphic"], "materials": "80% cotton, 20% polyester French terry"},
    {"title": "Slim Fit Chino Pants", "category": "casual", "price": 49.99,
     "desc": "Tailored slim-fit chinos in a stretch twill that moves with you. Sits at the natural waist with a tapered leg. Works equally well with sneakers or loafers.",
     "tags": ["chinos", "slim-fit", "stretch", "versatile"], "materials": "98% cotton, 2% elastane twill"},
    {"title": "Vintage Wash Denim Jacket", "category": "casual", "price": 79.99,
     "desc": "Classic trucker jacket in a medium-wash selvedge denim. Chest flap pockets, adjustable waist tabs, and copper-tone buttons. The perfect layering piece for spring and fall.",
     "tags": ["denim", "trucker", "vintage", "layering"], "materials": "100% selvedge cotton denim"},
    {"title": "Everyday Crossbody Bag", "category": "casual", "price": 34.99,
     "desc": "Compact crossbody bag with an adjustable strap, front zip pocket, and RFID-protected main compartment. Fits phone, wallet, keys, and a small water bottle. Weighs 8 oz.",
     "tags": ["crossbody", "RFID", "compact", "everyday"], "materials": "Water-resistant nylon, YKK zippers"},
    {"title": "Bamboo Fiber Crew Socks 6-Pack", "category": "casual", "price": 24.99,
     "desc": "Ultra-soft bamboo fiber socks with natural antibacterial and moisture-wicking properties. Reinforced heel and toe, seamless toe closure, and a mid-calf height. Fits sizes 6–12.",
     "tags": ["bamboo", "antibacterial", "moisture-wicking", "comfort"], "materials": "70% bamboo viscose, 25% cotton, 5% elastane"},
    {"title": "Relaxed Linen Drawstring Pants", "category": "casual", "price": 44.99,
     "desc": "Breezy linen pants with an elastic drawstring waist and side pockets. Perfect for beach vacations, weekend errands, or just lounging at home. Gets softer with every wash.",
     "tags": ["linen", "relaxed", "breathable", "vacation"], "materials": "100% pre-washed linen"},
    {"title": "Retro Aviator Sunglasses", "category": "casual", "price": 29.99,
     "desc": "Classic aviator frames in brushed gold metal with gradient smoke lenses. UV400 protection, adjustable nose pads, and spring-loaded temples for a comfortable fit.",
     "tags": ["aviator", "UV400", "retro", "metal-frame"], "materials": "Brushed metal alloy, gradient polycarbonate lenses"},

    # ── Health & Beauty ─────────────────────────────────────────────────────
    {"title": "SPF 50 Mineral Sunscreen", "category": "health", "price": 24.99,
     "desc": "Broad-spectrum mineral sunscreen that protects against UV rays without harmful chemicals. Reef-safe, non-greasy formula that absorbs quickly. Leaves no white cast.",
     "tags": ["sunscreen", "skincare", "SPF", "reef-safe"], "materials": "Zinc oxide, aloe vera"},
    {"title": "Hydrating Matte Lipstick", "category": "health", "price": 18.99,
     "desc": "Long-lasting matte lipstick infused with hyaluronic acid and shea butter for all-day comfort. Highly pigmented shade that doesn't feather or dry out your lips.",
     "tags": ["lipstick", "makeup", "beauty", "hydrating"], "materials": "Hyaluronic acid, shea butter, vegan pigments"},
    {"title": "Vitamin C Glow Serum", "category": "health", "price": 34.99,
     "desc": "Brightening facial serum with 15% Vitamin C and hyaluronic acid. Reduces dark spots, evens skin tone, and boosts collagen production. Safe for sensitive skin.",
     "tags": ["serum", "skincare", "vitamin-c", "anti-aging"], "materials": "Ascorbic acid, hyaluronic acid, vitamin E"},
    {"title": "Organic Lip Balm Trio", "category": "health", "price": 12.99,
     "desc": "Sustainably sourced beeswax lip balm set. Includes peppermint, vanilla, and unscented. Deeply moisturizes chapped lips during harsh weather.",
     "tags": ["lip-balm", "skincare", "organic", "moisturizing"], "materials": "Organic beeswax, coconut oil, essential oils"},
]


def _expand_catalog(templates: list[dict], target_count: int = 200) -> list[Product]:
    """
    Expand hand-authored templates into a full catalog by adding tasteful
    variations (color/version suffixes) while preserving description richness.
    """
    catalog: list[Product] = []
    color_variants = [
        "Charcoal", "Midnight Blue", "Forest Green", "Stone Grey",
        "Rust Orange", "Ivory", "Slate", "Obsidian",
    ]
    idx = 1
    variant_idx = 0

    while len(catalog) < target_count:
        for tmpl in templates:
            if len(catalog) >= target_count:
                break

            if variant_idx == 0:
                title = tmpl["title"]
            else:
                color = color_variants[variant_idx % len(color_variants)]
                title = f"{tmpl['title']} — {color}"

            catalog.append(Product(
                id=idx,
                title=title,
                category=tmpl["category"],
                price=round(tmpl["price"] * (1 + random.uniform(-0.08, 0.08)), 2),
                desc=tmpl["desc"],
                tags=list(tmpl["tags"]),
                rating=round(random.uniform(3.8, 5.0), 1),
                reviews=random.randint(12, 2400),
                materials=tmpl["materials"],
            ))
            idx += 1
        variant_idx += 1

    return catalog


def generate_catalog() -> list[Product]:
    """Generate the full product catalog."""
    return _expand_catalog(_TEMPLATES, target_count=len(_TEMPLATES))


def get_scenarios() -> dict[str, list[str]]:
    """
    Pre-built query sequences that demonstrate drift detection and
    self-healing adaptation in a recruiter demo.
    """
    return {
        "🟢 Phase 1 · Normal": [
            "I need a good water bottle for hiking.",
            "Looking for comfortable running shoes.",
            "Can you recommend a fitness watch with GPS?",
            "What kind of bags do you have for travel?",
        ],
        "🔴 Phase 2 · Black Friday": [
            "What's the absolute cheapest winter hat you have?",
            "Any bags under $25?",
            "Show me the most budget-friendly options.",
            "I only have $30 to spend, what can I get?",
        ],
        "☀️ Phase 3 · Summer Shift": [
            "Do you have lightweight sandals for the beach?",
            "I need breathable clothes for hot weather.",
            "Looking for UV protection sunglasses.",
            "Recommend summer vacation essentials.",
        ],
        "🌿 Phase 4 · Eco Trend": [
            "Show me products made from recycled materials.",
            "I only want sustainable, eco-friendly options.",
            "Do you have anything organic or plant-based?",
            "What's your most environmentally responsible product?",
        ],
    }
