"""Build the fine-tuning data: handbook questions answered in one fixed style.

Writes three files:
  blue_lantern_train.jsonl       - the model learns from these
  blue_lantern_validation.jsonl  - checked during training, never learned from
  blue_lantern_test.jsonl        - locked: only used to compare models after training

Run:  python make_dataset.py
"""

import json
import random
import re

random.seed(7)

# Split the handbook into sections: {"Membership Plans": "There are three plans...", ...}
with open("../document.txt", encoding="utf-8") as f:
    blocks = f.read().strip().split("\n\n")[1:]          # skip the title line
SECTIONS = {b.split("\n", 1)[0]: b.split("\n", 1)[1] for b in blocks}

UNKNOWN = ("**Answer:** I don't know. The handbook doesn't cover this.\n"
           "**Details:** Please ask the community manager, Farah Khan, at the front desk.\n"
           "**Source:** Not in the handbook")

# (section, question, answer, details). Section None = not in the handbook.
TRAIN = [
    # Welcome
    ("Welcome", "Where is Blue Lantern located?", "4th floor, Mayuri Towers, Madhapur, Hyderabad.", "It is a shared workspace in Hyderabad."),
    ("Welcome", "When was Blue Lantern founded?", "In 2021.", "It was founded by Kavya Reddy and Arjun Mehta."),
    ("Welcome", "How big is the space?", "120 desks, 6 meeting rooms and a 40-seat event hall.", "The event hall is called The Lighthouse."),
    ("Welcome", "what's the event hall called", "The Lighthouse.", "It seats 40 people."),

    # Opening Hours
    ("Opening Hours", "When are you open on Sundays?", "9:00 AM to 6:00 PM.", "Night Owl members have 24x7 access with their access card."),
    ("Opening Hours", "What time do you close on a Saturday?", "11:00 PM.", "The space is open 7:00 AM to 11:00 PM, Monday to Saturday."),
    ("Opening Hours", "What time do you open on weekdays?", "7:00 AM.", "The space is open 7:00 AM to 11:00 PM, Monday to Saturday."),
    ("Opening Hours", "Is the space open on Diwali?", "No.", "The space is closed on Diwali, Sankranti and Independence Day."),

    # Membership Plans
    ("Membership Plans", "How much is the Hot Desk plan?", "Rs. 4,500 per month.", "The Hot Desk plan gives you any free desk in the open area."),
    ("Membership Plans", "What does the Fixed Desk plan cost?", "Rs. 7,200 per month.", "You get your own desk with a lockable drawer."),
    ("Membership Plans", "Is there a security deposit?", "Yes, Rs. 3,000.", "All plans need this deposit, and it is refundable."),
    ("Membership Plans", "Do students get any discount?", "Yes, 20 percent on the Hot Desk plan.", "Show a valid college ID to get it."),
    ("Membership Plans", "I'm a student. Can I get a discount on the Night Owl plan?", "No.", "The student discount applies only to the Hot Desk plan."),
    ("Membership Plans", "How many plans do you have?", "Three.", "Hot Desk (Rs. 4,500), Fixed Desk (Rs. 7,200) and Night Owl (Rs. 9,000), all per month."),

    # Meeting Rooms
    ("Meeting Rooms", "How do I book a meeting room?", "Through the Lantern app.", "Every member gets 5 free meeting room hours per month."),
    ("Meeting Rooms", "How many free meeting room hours do I get?", "5 hours per month.", "After that, a room costs Rs. 300 per hour."),
    ("Meeting Rooms", "I've used my free hours. What does a room cost now?", "Rs. 300 per hour.", "Every member gets 5 free hours per month first."),
    ("Meeting Rooms", "Can I cancel a booking without paying?", "Yes, up to 2 hours before the start time.", "Meeting rooms are booked through the Lantern app."),

    # Internet and Printing
    ("Internet and Printing", "What's the WiFi called?", "BlueLantern-Members.", "The speed is 300 Mbps."),
    ("Internet and Printing", "How fast is the internet?", "300 Mbps.", "The WiFi network is called BlueLantern-Members."),
    ("Internet and Printing", "What is the WiFi password?", "It isn't in the handbook.", "The password changes on the first day of every month and is sent to members by email."),
    ("Internet and Printing", "How many pages can I print for free?", "50 black and white pages per month.", "Extra pages cost Rs. 2 per page."),

    # Guests
    ("Guests", "How many guests can I bring?", "Up to 2 guests per day.", "Guests must sign in at the front desk."),
    ("Guests", "How long can my guest stay?", "A maximum of 3 hours.", "Guests must sign in at the front desk."),
    ("Guests", "Do my visitors need to sign in?", "Yes, at the front desk.", "Members can bring up to 2 guests per day, for a maximum of 3 hours."),

    # Food and Pantry
    ("Food and Pantry", "Is the coffee free?", "Yes.", "Tea, coffee and filter water are free for all members."),
    ("Food and Pantry", "Can I get lunch there?", "Yes, on weekdays, for Rs. 120 per meal.", "Order it in the Lantern app before 10:30 AM."),
    ("Food and Pantry", "Can I order lunch on a Saturday?", "No.", "Catered lunch is available on weekdays only, for Rs. 120 per meal."),
    ("Food and Pantry", "Can I eat at my desk?", "Yes, except in the Quiet Zone.", "Please do not eat at the desks in the Quiet Zone."),

    # Pets
    ("Pets", "Can I bring my dog?", "Yes, on Fridays only.", "We call it Furry Friday. Dogs must stay on a leash."),
    ("Pets", "Can my dog come into a meeting room with me?", "No.", "Pets are not allowed in meeting rooms or the pantry."),
    ("Pets", "Are dogs allowed in the pantry?", "No.", "Pets are not allowed in meeting rooms or the pantry."),

    # Cancellation
    ("Cancellation", "How do I cancel my membership?", "Give 15 days of written notice.", "Email help@bluelantern.example."),
    ("Cancellation", "Which email do I write to when I want to leave?", "help@bluelantern.example.", "Give 15 days of written notice."),
    ("Cancellation", "When will I get my deposit back?", "Within 10 working days after you return your access card.", "The deposit is Rs. 3,000."),

    # Support
    ("Support", "Who is the community manager?", "Farah Khan.", "She is at the front desk from 9:00 AM to 6:00 PM on weekdays."),
    ("Support", "When is Farah Khan at the front desk?", "9:00 AM to 6:00 PM on weekdays.", "She is the community manager."),

    # Not in the handbook: always the same "I don't know" reply
    (None, "Is there parking?", None, None),
    (None, "Who is the CEO of Google?", None, None),
    (None, "Do you sell day passes?", None, None),
    (None, "How much does a locker cost?", None, None),
    (None, "Can I host a birthday party here?", None, None),
    (None, "Is there air conditioning?", None, None),
    (None, "Can I store my bicycle inside?", None, None),
    (None, "Do you have a nap room?", None, None),
    (None, "Is smoking allowed on the terrace?", None, None),
]

# Checked during training (validation loss); the model never learns from these.
VALIDATION = [
    ("Membership Plans", "Night Owl price?", "Rs. 9,000 per month.", "It includes a fixed desk plus 24x7 access."),
    ("Food and Pantry", "When is the pantry restocked?", "Every morning at 8:00 AM.", "Tea, coffee and filter water are free."),
    ("Welcome", "Who started Blue Lantern?", "Kavya Reddy and Arjun Mehta.", "They founded it in 2021."),
    ("Cancellation", "I lost my access card. What now?", "A replacement costs Rs. 500.", "You need to return your access card to get your security deposit back when you leave."),
    (None, "Is there a gym?", None, None),
]

# LOCKED test set: never used for training or tuning, only to compare models.
# (section, question, words the answer must contain to be factually correct)
# Each test question gets a context built the same way as the training examples.
TEST = [
    ("Opening Hours", "I like working late. Can I come in at 2 AM?", ["Night Owl"]),
    ("Membership Plans", "Which plan should I pick if I want my own desk?", ["Fixed Desk"]),
    ("Meeting Rooms", "Which room fits a team of 12?", ["Deep Sea"]),
    ("Internet and Printing", "How much is a colour print?", ["10"]),
    ("Guests", "My friend wants to visit after work, around 9 PM. Is that ok?", ["8"]),
    ("Pets", "Can I bring my cat?", ["dog"]),
    ("Support", "Who do I call for an urgent problem at night?", ["104"]),
    (None, "Is the building wheelchair accessible?", ["don't know"]),
    (None, "Do you have showers?", ["don't know"]),
]


def prompt(context, question):
    """The exact prompt app.py builds in 'Chat with a document' mode."""
    return ("Answer the question using only the context below. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}")


def context_for(section):
    """The right section plus one or two unrelated ones, like real retrieval returns."""
    others = [s for s in SECTIONS if s != section]
    picked = random.sample(others, 2 if section is None else 1)
    if section:
        picked.append(section)
    random.shuffle(picked)
    return "\n\n---\n\n".join(f"{s}\n{SECTIONS[s]}" for s in picked)


def write_conversations(path, examples):
    # newline="\n" gives the same file, byte for byte, on Windows, macOS and Linux.
    with open(path, "w", encoding="utf-8", newline="\n") as out:
        for section, question, answer, details in examples:
            if section:
                assert section in SECTIONS, f"No section called {section!r}"
                reply = f"**Answer:** {answer}\n**Details:** {details}\n**Source:** {section}"
            else:
                reply = UNKNOWN
            row = {"conversations": [
                {"role": "user", "content": prompt(context_for(section), question)},
                {"role": "assistant", "content": reply},
            ]}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


# --- Leakage check: no test question may be a near-copy of a training one ---------

STOPWORDS = {"a", "an", "the", "is", "are", "do", "does", "i", "my", "me", "can", "you",
             "to", "of", "in", "on", "at", "for", "and", "or", "what", "how", "which",
             "who", "when", "there", "it", "if", "want", "much", "many", "s", "what's"}


def words(text):
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOPWORDS}


def overlap(a, b):
    """Share of meaningful words two questions have in common (0 = none, 1 = identical)."""
    wa, wb = words(a), words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


seen = [q for _, q, _, _ in TRAIN + VALIDATION]
assert len(set(seen)) == len(seen), "Duplicate question in train/validation"
for _, question, _ in TEST:
    closest = max(seen, key=lambda q: overlap(question, q))
    score = overlap(question, closest)
    assert score < 0.5, f"Test question leaks: {question!r} is too close to {closest!r} ({score:.2f})"

# --- Write the three files ------------------------------------------------------

write_conversations("blue_lantern_train.jsonl", TRAIN)
write_conversations("blue_lantern_validation.jsonl", VALIDATION)

with open("blue_lantern_test.jsonl", "w", encoding="utf-8", newline="\n") as out:
    for section, question, must_contain in TEST:
        assert section is None or section in SECTIONS, f"No section called {section!r}"
        row = {"question": question,
               "context": context_for(section),      # the same shape as training and the app
               "source": section or "Not in the handbook",
               "must_contain": must_contain}
        out.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Train:      {len(TRAIN)} examples -> blue_lantern_train.jsonl")
print(f"Validation: {len(VALIDATION)} examples -> blue_lantern_validation.jsonl")
print(f"Test:       {len(TEST)} questions -> blue_lantern_test.jsonl (locked)")
print("Leakage check passed: no test question is close to a training or validation one.")
