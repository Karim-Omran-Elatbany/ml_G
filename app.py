from flask import Flask, request, render_template, url_for, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

def preprossing(image):
    image=Image.open(image)
    image = image.resize((224, 224))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (1, 224, 224, 3)
    return image_arr
classes = ['Akhenaten', 'Bent pyramid for senefru', 'Colossal Statue of Ramesses II', 'Colossoi of Memnon', 'Goddess Isis with her child', 'Hatshepsut', 'Khafre Pyramid', 'King Thutmose III', 'Mask of Tutankhamun', 'Nefertiti', 'Pyramid_of_Djoser', 'Ramessum', 'Statue of King Zoser', 'Statue of Tutankhamun with Ankhesenamun', 'Temple_of_Isis_in_Philae', 'Temple_of_Kom_Ombo', 'The Great Temple of Ramesses II', 'amenhotep iii and tiye', 'bust of ramesses ii', 'head Statue of Amenhotep iii', 'menkaure pyramid', 'sphinx']
class_texts = {
                  'Akhenaten': """
    Akhenaten, also known as Amenhotep IV, was an ancient Egyptian pharaoh who ruled during the 18th dynasty, around 1353-1336 BCE. He is best known for his attempt to revolutionize Egyptian religion and society by introducing a monotheistic cult centered around the worship of the sun disc, Aten. 
    During his reign, he moved the capital to a new city called Akhetaten (modern-day Amarna), and he commissioned a series of radical changes in art and ideology. The art of this period is characterized by a departure from traditional Egyptian artistic conventions, with depictions of the royal family in more naturalistic and intimate settings.
    Akhenaten's religious reforms included the suppression of the traditional polytheistic cults and the elevation of the Aten to the status of the supreme god. This was accompanied by the eradication of references to other gods and the centralization of religious authority under the pharaoh.
    However, Akhenaten's reforms were largely unpopular, and after his death, his successors worked to restore the old religious order. Akhenaten himself was largely forgotten until modern archaeological discoveries brought his reign to light, sparking renewed interest in his reign and its implications for ancient Egyptian history and religion.
    """,
    'Bent Pyramid for Sneferu': """
    The Bent Pyramid is an ancient Egyptian pyramid located in the Dahshur necropolis, built during the Old Kingdom period by Pharaoh Sneferu, who reigned around 2600 BCE. It represents a transition in pyramid design, showcasing experimentation with different architectural techniques.
    The pyramid is named for its unusual bent shape, resulting from an alteration in its angle of inclination during construction. Initially, the pyramid was built with a steep angle, likely due to concerns about structural stability. However, as construction progressed, it became evident that the steep angle could lead to structural failure. Consequently, the angle was reduced, resulting in the distinctive bent appearance.
    Despite its unconventional shape, the Bent Pyramid is significant for its role in the development of pyramid construction techniques. It served as a precursor to the later, more successful pyramids built by Sneferu's successors, including the Red Pyramid, also located in Dahshur.
    The Bent Pyramid is part of Sneferu's extensive building projects, reflecting the wealth and power of the Old Kingdom Egyptian state. It stands as a testament to the ingenuity and architectural prowess of the ancient Egyptians, showcasing their ability to adapt and innovate in the pursuit of monumental construction projects.
    """,
    'Colossal Statue of Ramesses II': """
    The Colossal Statue of Ramesses II is an iconic ancient Egyptian sculpture that depicts one of the most celebrated pharaohs of the New Kingdom, Ramesses II, also known as Ramesses the Great. Carved from a single block of limestone, the statue stands over 10 meters (33 feet) tall and weighs around 83 tons.
    Originally located in the Ramesseum temple complex in Thebes (modern-day Luxor), the statue portrays Ramesses II in a seated position, wearing the traditional royal regalia of the pharaoh. His hands are placed on his knees, and he wears the double crown of Upper and Lower Egypt, symbolizing his rule over the unified kingdom.
    The Colossal Statue of Ramesses II served both religious and political purposes. As a monumental depiction of the pharaoh, it was intended to inspire awe and reverence among the ancient Egyptians, emphasizing Ramesses II's divine and kingly status. It also served as a testament to the might and power of the Egyptian state during the New Kingdom period.
    The statue's colossal size and intricate craftsmanship reflect the skill and artistry of ancient Egyptian sculptors and artisans. Despite being over three millennia old, the statue has survived relatively intact, attesting to the durability of ancient Egyptian monumental sculpture.
    Today, the Colossal Statue of Ramesses II is one of the most famous and recognizable artifacts from ancient Egypt. It has been relocated several times throughout history, and it is currently housed in the Grand Egyptian Museum in Giza, where it continues to attract visitors from around the world, serving as a tangible link to Egypt's rich cultural heritage.
    """,
    'Colossi of Memnon': """
    The Colossi of Memnon are two massive stone statues located on the west bank of the Nile River in Luxor, Egypt. These statues depict Pharaoh Amenhotep III, who reigned during the 18th dynasty of ancient Egypt, around 1350 BCE.
    Originally, the Colossi of Memnon stood at the entrance of Amenhotep III's mortuary temple, known as the "Temple of Millions of Years." The statues were constructed from quartzite sandstone and stood about 18 meters (60 feet) tall. Each statue depicted the pharaoh seated on a throne, with his hands resting on his knees and wearing the royal regalia.
    Over time, the Colossi of Memnon became famous for a unique phenomenon. At dawn, when the first rays of sunlight struck the statues, they emitted a mysterious sound resembling the sound of a human voice. Ancient Greek travelers visiting the site attributed this phenomenon to the mythical figure Memnon, son of the dawn goddess Eos, mourning for his fallen comrades during the Trojan War.
    The Colossi of Memnon suffered damage and erosion over the centuries, with one of the statues collapsing in antiquity. Today, only the northern statue remains largely intact, while the southern statue is largely in ruins.
    Despite their partial destruction, the Colossi of Memnon remain significant archaeological and tourist attractions in Luxor. They stand as enduring symbols of the grandeur and power of ancient Egyptian civilization, attracting visitors from around the world who come to marvel at their imposing presence and learn about their rich historical and cultural significance.
    """,
    'Goddess Isis with her child': """
    The Goddess Isis with her child is an ancient Egyptian religious motif representing the goddess Isis, who was revered as a powerful deity associated with motherhood, magic, and fertility, cradling her infant son, Horus. In this depiction, Isis is typically shown nursing or protecting her child, Horus, who later became one of the most important gods in the Egyptian pantheon.
    This motif holds significant religious and cultural significance in ancient Egyptian mythology, symbolizing the divine maternal bond between Isis and Horus, as well as the eternal cycle of life, death, and rebirth. It also reflects broader themes of protection, nurturing, and the role of motherhood in Egyptian society.
    The image of Isis with her child has persisted throughout history and has been depicted in various forms of ancient Egyptian art, including sculptures, reliefs, and amulets. It continues to be a subject of fascination and reverence in modern times, both within Egypt and around the world, as a symbol of maternal love, protection, and the enduring power of the divine feminine.
    """,
    'Hatshepsut': """
    Hatshepsut was one of ancient Egypt's most notable and unique pharaohs, reigning during the 18th dynasty around the 15th century BCE. She is particularly renowned for being one of the few women to rule as pharaoh in ancient Egypt and for the significant architectural and trade expeditions she undertook during her reign.
    Hatshepsut was the daughter of Thutmose I and became queen when she married her half-brother, Thutmose II. Upon his death, she assumed the role of regent for her stepson, Thutmose III, but eventually declared herself pharaoh, effectively ruling as king of Egypt. She depicted herself as a male pharaoh in many statues and reliefs, wearing the traditional pharaonic regalia, including the false beard.
    During her reign, Hatshepsut initiated numerous building projects, including the construction of the spectacular mortuary temple at Deir el-Bahri near Thebes. This temple, known as Djeser-Djeseru or "Holy of Holies," is considered one of the most beautiful temples in Egypt and is a testament to her architectural prowess and grand vision.
    Hatshepsut also facilitated trade expeditions to the land of Punt, a region rich in exotic goods such as gold, ivory, ebony, and incense. These expeditions not only brought valuable resources to Egypt but also helped to strengthen diplomatic and cultural ties with other civilizations.
    Despite her remarkable achievements, Hatshepsut's legacy was partially overshadowed by later pharaohs who attempted to erase her from history. Many of her statues were defaced or destroyed, and her name was omitted from official records. However, in modern times, she has been rediscovered and celebrated as one of ancient Egypt's most successful and influential rulers, breaking gender barriers and leaving behind a lasting legacy of architectural splendor and diplomatic achievement
    """,
    'Khafre Pyramid': """
    The Khafre Pyramid is one of the most iconic structures of ancient Egypt, located on the Giza Plateau near Cairo. It was built during the Fourth Dynasty by Pharaoh Khafre, who reigned around 2558–2532 BCE.
    This pyramid is the second-largest of the Giza pyramids, standing at approximately 136 meters (446 feet) in height. Originally, it was clad in polished Tura limestone, which gave it a smooth and gleaming appearance, although much of this casing has eroded over time.
    Like other pyramids of the era, the Khafre Pyramid served as a royal tomb, housing the mummified remains of Pharaoh Khafre. Its interior consists of a series of passages and chambers, including the burial chamber where the sarcophagus would have been placed.
    One of the distinctive features of the Khafre Pyramid is the presence of the Great Sphinx, a massive limestone statue with the body of a lion and the head of a human, believed to represent Khafre himself. The Sphinx sits near the pyramid's eastern face, guarding its entrance.
    The Khafre Pyramid is noted for its slightly smaller size compared to the Great Pyramid of Giza, built by Khafre's father, Khufu. However, it is often considered to be more aesthetically pleasing due to its better preservation and the remnants of its original casing stones.
    Overall, the Khafre Pyramid stands as a testament to the engineering and architectural achievements of ancient Egypt, showcasing the ingenuity and craftsmanship of the civilization that built it. It remains a popular tourist attraction and a symbol of Egypt's rich cultural heritage.
    """,
    'King Thutmose III': """
    Thutmose III, also known as Thutmose the Great, was one of ancient Egypt's most accomplished pharaohs, ruling during the New Kingdom period, around the 15th century BCE. He is remembered for his military conquests, administrative reforms, and cultural advancements, earning him a reputation as one of Egypt's most successful warrior kings.
    Thutmose III ascended to the throne after the death of his stepmother, Hatshepsut, who had acted as his regent during his early years. Initially, he shared power with his co-regent, but he eventually assumed full control and embarked on a series of military campaigns to expand Egypt's territory and influence.
    Thutmose III conducted numerous military campaigns across the Levant, Nubia, and into Mesopotamia, consolidating Egypt's control over these regions and establishing the empire's dominance in the Near East. His most famous military victory came at the Battle of Megiddo, where he defeated a coalition of Canaanite city-states, securing Egypt's control over the lucrative trade routes of the eastern Mediterranean.
    In addition to his military achievements, Thutmose III implemented administrative reforms that strengthened central authority, reorganized the bureaucracy, and promoted economic development. He also fostered cultural and religious advancements, commissioning temples, monuments, and religious festivals to honor the gods and legitimize his rule.
    Thutmose III's reign marked a golden age for ancient Egypt, characterized by prosperity, stability, and imperial expansion. His military conquests and administrative reforms solidified Egypt's position as a dominant power in the ancient world and left a lasting legacy of achievement and innovation. He is remembered as one of Egypt's greatest pharaohs, earning him the epithet "Napoleon of Egypt" among modern historians.
    """,
    'Mask of Tutankhamun': """
    The Mask of Tutankhamun is one of the most iconic and recognizable artifacts from ancient Egypt. It is a funerary mask crafted from gold and precious stones that was found covering the head of the mummy of the pharaoh Tutankhamun in his tomb in the Valley of the Kings near Luxor.
    Crafted during the 18th dynasty, around 1323 BCE, the mask is an exquisite example of ancient Egyptian artistry and craftsmanship. It is made of solid gold, weighing approximately 11 kilograms (24 pounds), and is adorned with inlaid semi-precious stones such as lapis lazuli, turquoise, and carnelian.
    The mask portrays the youthful features of Tutankhamun, with delicate facial features and almond-shaped eyes. It is surmounted by a nemes headcloth, a striped headdress worn by pharaohs, and features a ceremonial beard attached by gold wires.
    The Mask of Tutankhamun served both a practical and symbolic purpose. It protected the pharaoh's mummy and was believed to help guide his spirit into the afterlife. Additionally, it symbolized Tutankhamun's divinity and status as a god-king, serving as a powerful symbol of royal authority and immortality.
    The discovery of Tutankhamun's tomb by Howard Carter in 1922 sparked worldwide fascination with ancient Egypt and its treasures, with the Mask of Tutankhamun becoming an enduring symbol of that era. Today, it is one of the most famous artifacts in the world and is housed in the Egyptian Museum in Cairo, where it continues to captivate visitors with its beauty and historical significance.
    """,
    'Nefertiti': """
    Nefertiti was one of ancient Egypt's most renowned queens, celebrated for her beauty, influence, and prominence during the 14th century BCE. She was the wife of Pharaoh Akhenaten, who initiated a religious revolution in ancient Egypt by introducing the worship of the sun god, Aten, as the supreme deity.
    Nefertiti played a significant role in her husband's religious reforms, and she was depicted alongside him in numerous reliefs and statues, suggesting that she held considerable power and influence in the royal court. She is often depicted as an idealized beauty with a graceful neck, elegantly sculpted features, and an enigmatic smile.
    One of the most famous depictions of Nefertiti is the bust discovered in Amarna, the capital city founded by Akhenaten. This iconic sculpture, now housed in the Neues Museum in Berlin, Germany, has become a symbol of ancient Egyptian artistry and feminine beauty.
    Nefertiti's exact origins and fate remain shrouded in mystery. Some scholars believe she was of Egyptian royal blood, while others suggest she may have been a foreign princess. The circumstances of her death and the identity of her successor as queen consort are also subject to debate and speculation.
    Despite these uncertainties, Nefertiti's legacy endures as one of ancient Egypt's most enigmatic and influential figures. Her beauty, grace, and power have captured the imagination of scholars, artists, and admirers throughout history, making her one of the most iconic and celebrated queens of ancient Egypt.
    """, 
    'Pyramid_of_Djoser': """
    The Pyramid of Djoser, also known as the Step Pyramid, is an ancient Egyptian monument located in the Saqqara necropolis near Cairo. Built during the 27th century BCE, it was commissioned by Pharaoh Djoser, who ruled during the Third Dynasty of the Old Kingdom period.
    Designed by the architect Imhotep, the Pyramid of Djoser represents a significant innovation in ancient Egyptian architecture, marking the transition from simple mastaba tombs to more elaborate pyramid structures. It is considered the earliest colossal stone building in Egypt and the first pyramid ever constructed.
    The pyramid originally stood at approximately 62 meters (203 feet) tall, consisting of six stepped layers or "mastabas" stacked on top of each other. These layers were constructed using limestone blocks, with underground tunnels and chambers serving as the pharaoh's burial complex.
    Surrounding the pyramid is a vast mortuary complex, including courtyards, temples, shrines, and galleries. These structures were dedicated to the cult of the deceased pharaoh and served religious and ceremonial purposes.
    The Pyramid of Djoser represents not only a monumental architectural achievement but also a significant cultural and religious development in ancient Egypt. It reflects the growing power and authority of the pharaohs, as well as their evolving beliefs about the afterlife and the role of monumental architecture in ensuring their eternal existence.
    Today, the Pyramid of Djoser stands as a UNESCO World Heritage Site and a symbol of ancient Egypt's rich cultural heritage. It continues to inspire awe and fascination among visitors, offering insights into the ingenuity and craftsmanship of the ancient Egyptians and the enduring legacy of their civilization.
    """, 
    'Ramessum': """
    The Ramesseum is an ancient Egyptian temple complex located on the west bank of the Nile River near Luxor. It was built during the 13th century BCE by Pharaoh Ramesses II, also known as Ramesses the Great, to serve as a mortuary temple dedicated to the worship of the god Amun and the commemoration of the pharaoh's reign and accomplishments.
    The Ramesseum is one of the largest and most well-preserved temple complexes in Egypt, showcasing the grandeur and architectural prowess of the New Kingdom period. It consists of a series of courtyards, halls, pylons, and chapels surrounded by massive mudbrick enclosure walls.
    One of the most notable features of the Ramesseum is its colossal seated statue of Ramesses II, which originally stood at the entrance of the temple complex. Although much of the statue has been damaged over time, its colossal size and imposing presence are a testament to the pharaoh's power and divine status.
    The walls of the Ramesseum are adorned with intricate reliefs and hieroglyphic inscriptions depicting scenes from Ramesses II's military campaigns, religious rituals, and divine triumphs. These reliefs provide valuable insights into ancient Egyptian history, religion, and culture.
    Despite suffering damage from natural disasters and human intervention over the centuries, the Ramesseum remains a significant archaeological and cultural site in Egypt. It attracts visitors from around the world who come to marvel at its impressive architecture, learn about ancient Egyptian civilization, and explore the legacy of one of Egypt's most renowned pharaohs
    """, 
    'Statue of King Zoser': """
    The Statue of King Djoser is an ancient Egyptian sculpture that dates back to the Third Dynasty of the Old Kingdom period, around 2700 BCE. It is one of the earliest known life-size royal statues in Egyptian history and is associated with Pharaoh Djoser, who commissioned the construction of the Step Pyramid at Saqqara, near modern-day Cairo.
    This limestone statue portrays Pharaoh Djoser in a seated position, with his hands placed flat on his knees, a common pose in ancient Egyptian royal statuary. It is believed to have been originally located in Djoser's mortuary complex at Saqqara, where it served a religious and ceremonial function, possibly as part of the king's cult worship after his death.
    The statue is notable for its relatively naturalistic portrayal of the pharaoh, with details such as facial features and musculature depicted with a degree of realism. This departure from earlier, more stylized representations reflects the artistic and cultural developments of the early Old Kingdom period.
    The Statue of King Djoser is also significant because it provides valuable insights into the political, religious, and artistic aspects of ancient Egyptian society during the Early Dynastic period. It serves as a tangible link to Egypt's ancient past and highlights the importance of royal cult worship and funerary rituals in ancient Egyptian religion.
    Today, the Statue of King Djoser is housed in the Egyptian Museum in Cairo, where it continues to be studied and admired by scholars, historians, and visitors alike, offering a glimpse into the grandeur and sophistication of ancient Egyptian civilization
    """,
    'Statue of Tutankhamun with Ankhesenamun':"""
    There isn't a specific well-known statue of Tutankhamun with Ankhesenamun that matches the same level of fame as, for example, the Statue of King Djoser or the Colossi of Memnon. However, Tutankhamun and Ankhesenamun were historically notable figures as they were both members of the royal family during the 18th dynasty of ancient Egypt.
    Tutankhamun, often referred to as the "boy king," ruled during the New Kingdom period around 1332–1323 BCE. He is most famous for the discovery of his nearly intact tomb in the Valley of the Kings in 1922, which contained a wealth of treasures and artifacts that provided valuable insights into ancient Egyptian culture and burial practices.
    Ankhesenamun, on the other hand, was Tutankhamun's half-sister and wife. She is known for her significant role in the royal family and the political intrigues of the time, particularly following Tutankhamun's death. After his demise, Ankhesenamun is believed to have married the pharaoh Ay, possibly her grandfather, in an attempt to maintain her status and influence.
    While there may not be a specific statue depicting Tutankhamun and Ankhesenamun together that stands out in the same way as other ancient Egyptian sculptures, their relationship and historical significance continue to capture the imagination of scholars and enthusiasts alike. Their story is a fascinating chapter in the rich tapestry of ancient Egyptian history and royal lineage.
    """,
    'Temple_of_Isis_in_Philae':"""
    The Temple of Isis in Philae is an ancient Egyptian temple complex dedicated to the goddess Isis, located on the island of Philae in the Nile River near Aswan, Egypt. Constructed during the Ptolemaic and Roman periods, it served as one of the most important religious centers in ancient Egypt and continued to be used for worship for centuries.
    The temple complex at Philae consists of a series of structures, including the main temple dedicated to Isis, smaller chapels, gateways, and pylons. The main temple is adorned with intricate reliefs, carvings, and hieroglyphic inscriptions that depict scenes from ancient Egyptian mythology, religious rituals, and the cult of Isis.
    The Temple of Isis was renowned throughout the ancient world as a center of pilgrimage and worship, attracting devotees from across Egypt and beyond who sought the blessings and protection of the goddess Isis. It was also the site of important religious festivals and ceremonies, including the annual celebration of the Osiris Mysteries, which commemorated the myth of Isis and Osiris.
    In addition to its religious significance, the Temple of Isis is also renowned for its architectural beauty and picturesque setting on the island of Philae, surrounded by the waters of the Nile. The temple's location on an island also helped to preserve its structures from the flooding caused by the construction of the Aswan High Dam in the 20th century.
    Today, the Temple of Isis at Philae is a UNESCO World Heritage Site and a popular tourist destination in Egypt. Despite being relocated from its original island to nearby Agilkia Island to avoid submersion by the rising waters of Lake Nasser, it continues to attract visitors who come to marvel at its stunning architecture, learn about ancient Egyptian religion and mythology, and experience the timeless beauty of one of Egypt's most sacred sites.
    """, 
    'Temple_of_Kom_Ombo':"""
    The Temple of Kom Ombo is an ancient Egyptian temple located in the town of Kom Ombo, near Aswan, Egypt. It is unique among Egyptian temples because it is dedicated to two deities, Sobek, the crocodile god, and Horus the Elder, the falcon-headed god.
    Built during the Ptolemaic dynasty, around the 2nd century BCE, the Temple of Kom Ombo is situated on a picturesque setting overlooking the Nile River. It consists of two symmetrical halves, each dedicated to a different deity. This dual dedication is reflected in the temple's layout, with twin entrances, halls, sanctuaries, and chapels for each god.
    The temple is adorned with intricately carved reliefs, columns, and hieroglyphic inscriptions that depict various scenes from ancient Egyptian mythology, religious rituals, and offerings to the gods. One of the most notable features of the temple is the Nilometer, a device used to measure the water level of the Nile during flood season, which was crucial for determining taxes and agricultural planning.
    The Temple of Kom Ombo was a significant religious center in ancient Egypt, attracting pilgrims and devotees who sought the protection and blessings of the gods Sobek and Horus. It was also the site of important religious festivals and ceremonies, including the annual Feast of Sobek, during which live crocodiles were worshipped and mummified as offerings to the god.
    Today, the Temple of Kom Ombo is a popular tourist attraction in Egypt, drawing visitors from around the world who come to admire its unique architecture, learn about ancient Egyptian religion and history, and experience the timeless beauty of one of Egypt's most fascinating archaeological sites.
    """, 
   'The Great Temple of Ramesses II': """
   The Great Temple of Ramesses II, also known as the Temple of Abu Simbel, is one of ancient Egypt's most magnificent and well-known monuments. It was built during the reign of Pharaoh Ramesses II in the 13th century BCE, in the southern part of Egypt near the border with Sudan.
    Carved directly into the sandstone cliffs on the western bank of the Nile River, the temple complex consists of two main temples: the Great Temple dedicated to the gods Ra-Horakhty, Ptah, and Amun, and the smaller Temple of Hathor dedicated to Ramesses II's beloved wife, Queen Nefertari.
    The Great Temple is the larger of the two and is famous for its colossal statues of Ramesses II, each standing over 20 meters (65 feet) tall. The facade of the temple features four seated statues of the pharaoh, each wearing the double crown of Upper and Lower Egypt. The interior of the temple is adorned with intricate reliefs and hieroglyphic inscriptions that depict scenes from Ramesses II's military campaigns, religious rituals, and divine triumphs.
    One of the most remarkable aspects of the Great Temple of Ramesses II is its alignment with the sun. Twice a year, on February 22nd and October 22nd, the sun's rays penetrate the temple's inner sanctum and illuminate the statues of the gods seated inside, including a statue of Ramesses II himself, while leaving the statue of the god Ptah in shadow.
    The Great Temple of Ramesses II is a UNESCO World Heritage Site and a symbol of ancient Egypt's grandeur and cultural heritage. It continues to awe and inspire visitors from around the world with its monumental architecture, artistic beauty, and timeless significance
    """,
    'amenhotep iii and tiye': """
    Amenhotep III and Queen Tiye were prominent figures in ancient Egypt during the 18th dynasty of the New Kingdom period, around the 14th century BCE. Amenhotep III, also known as Amenhotep the Magnificent, was one of Egypt's most powerful and prosperous pharaohs, known for his extensive building projects, diplomatic achievements, and cultural advancements.
    During his nearly four-decade-long reign, Amenhotep III oversaw the construction of numerous temples, monuments, and statues throughout Egypt, including the impressive mortuary temple known as the "Temple of Millions of Years" at Luxor. He also maintained close diplomatic relations with foreign powers, particularly with the Mitanni kingdom in Mesopotamia and the Hittite Empire in Anatolia, fostering trade and political alliances.
    Queen Tiye, Amenhotep III's principal wife, played a significant role in the royal court and is believed to have exerted considerable influence over her husband. She was highly respected and revered, often depicted alongside the pharaoh in reliefs and statues, a departure from traditional depictions of queens in ancient Egypt.
    Tiye was also known for her intelligence, political acumen, and diplomatic skills, and she played a key role in negotiating alliances and diplomatic marriages between Egypt and foreign powers. She was a prominent figure in both domestic and international affairs, and her influence extended beyond the royal court to shape the course of Egyptian history.
    Together, Amenhotep III and Queen Tiye presided over a period of unprecedented prosperity and cultural flourishing in ancient Egypt, known as the "Amarna Period." Their reign saw the height of Egypt's power and influence, marked by unparalleled wealth, artistic achievement, and diplomatic success.
    Despite their significant contributions to ancient Egyptian civilization, Amenhotep III and Queen Tiye are perhaps best remembered as the parents of one of Egypt's most famous pharaohs, Akhenaten, who would go on to initiate a religious revolution that would reshape Egyptian society and religion
    """,
    'bust of ramesses ii':"""
    The Bust of Ramesses II is a famous ancient Egyptian sculpture depicting one of the most renowned pharaohs of Egypt's New Kingdom period. Ramesses II, also known as Ramesses the Great, ruled during the 19th dynasty, around the 13th century BCE.
    This limestone bust portrays Ramesses II in a regal and dignified manner, with finely sculpted facial features, a pronounced brow, and a powerful jawline. It is believed to have been originally part of a larger statue, likely situated in one of Egypt's temples or royal complexes.
    The bust of Ramesses II is notable for its exquisite craftsmanship and attention to detail, reflecting the skill and artistry of ancient Egyptian sculptors. It captures the pharaoh's likeness with remarkable accuracy, conveying a sense of authority and divine kingship.
    Ramesses II was one of ancient Egypt's most prolific builders, commissioning numerous temples, monuments, and statues throughout the kingdom. He is perhaps best known for his grand architectural projects, including the construction of the Great Temple of Abu Simbel and the Ramesseum in Thebes.
    Today, the Bust of Ramesses II serves as a testament to the enduring legacy of one of Egypt's greatest pharaohs. It is housed in various museums around the world, where it continues to captivate visitors with its timeless beauty and historical significance, offering a tangible connection to the splendor of ancient Egyptian civilization.
    """,
    'head Statue of Amenhotep iii':"""
    The head statue of Amenhotep III is an ancient Egyptian sculpture depicting one of the most prominent pharaohs of the New Kingdom period. Amenhotep III, also known as Amenhotep the Magnificent, reigned during the 18th dynasty, around the 14th century BCE.
    This head statue, crafted from finely carved stone, portrays Amenhotep III with striking realism and attention to detail. The pharaoh's facial features are depicted with precision, including his distinctive almond-shaped eyes, prominent nose, and full lips. The sculpture conveys a sense of regal authority and divine kingship, characteristic of ancient Egyptian royal portraiture.
    Amenhotep III was one of ancient Egypt's most prosperous and powerful pharaohs, known for his extensive building projects, diplomatic achievements, and cultural patronage. During his reign, Egypt experienced a period of unprecedented wealth and stability, marked by the construction of magnificent temples, monuments, and statues throughout the kingdom.
    The head statue of Amenhotep III is a testament to the artistic skill and craftsmanship of ancient Egyptian sculptors, who were able to capture the essence of the pharaoh's likeness with remarkable accuracy. It serves as a tangible link to the splendor of ancient Egyptian civilization and the enduring legacy of one of its greatest rulers.
    Today, the head statue of Amenhotep III is housed in various museums around the world, where it continues to be admired and studied by scholars, historians, and visitors alike, offering insights into the art, culture, and history of ancient Egypt.
    """, 
    'menkaure pyramid': """
    The Pyramid of Menkaure, also known as Menkaure's Pyramid or the Pyramid of Mykerinos, is an ancient Egyptian pyramid located on the Giza Plateau near Cairo, Egypt. It was built during the Fourth Dynasty of the Old Kingdom period, around 2510–2490 BCE, and is one of the three main pyramids at the Giza complex, along with the Great Pyramid of Khufu and the Pyramid of Khafre.
    The Pyramid of Menkaure is the smallest of the three pyramids at Giza, standing at approximately 65 meters (213 feet) in height. It is constructed of limestone blocks, which were originally encased in polished Tura limestone, giving it a smooth and gleaming appearance.
    The pyramid was built as a tomb for Pharaoh Menkaure (also known as Mykerinos), who was the fifth ruler of the Fourth Dynasty. The burial chamber, located deep within the pyramid's core, housed the pharaoh's sarcophagus and funerary goods intended for the afterlife.
    Surrounding the pyramid are several subsidiary structures, including three small pyramids known as "queen's pyramids," as well as temples and mastaba tombs for members of the royal family and high-ranking officials.
    Despite its smaller size, the Pyramid of Menkaure is notable for its architectural features, including the precision of its construction and the quality of its masonry. It reflects the technological and engineering achievements of ancient Egyptian civilization and serves as a testament to the power and authority of the pharaohs of the Old Kingdom.
    Today, the Pyramid of Menkaure stands as a symbol of ancient Egypt's rich cultural heritage and continues to attract visitors from around the world who come to marvel at its monumental architecture and learn about the history and significance of one of the world's most famous archaeological sites.
    """,
    'sphinx': """
    The Sphinx is an iconic ancient Egyptian monument located on the Giza Plateau near Cairo, Egypt. Carved from a single limestone outcrop, it is one of the largest and most recognizable statues in the world, measuring approximately 73 meters (240 feet) in length and standing about 20 meters (66 feet) tall.
    The Sphinx is believed to have been constructed during the reign of the pharaoh Khafre, who ruled during the Fourth Dynasty of the Old Kingdom period, around 2500 BCE. It is commonly thought to represent the pharaoh Khafre himself, although some theories suggest that it may depict other individuals or mythical figures.
    The statue has the body of a lion, symbolizing strength and power, and the head of a human, typically believed to represent the divine authority of the pharaoh. The face of the Sphinx bears the features of a pharaoh, with a serene expression and a headdress similar to that worn by the pharaohs of ancient Egypt.
    The Sphinx is surrounded by mystery and legend, with numerous myths and theories surrounding its construction, purpose, and symbolism. It has been the subject of speculation and fascination for centuries, inspiring countless works of art, literature, and popular culture.
    Despite being weathered by time and erosion, the Sphinx remains one of Egypt's most iconic and enduring symbols, attracting millions of visitors from around the world who come to marvel at its monumental presence and learn about its rich history and significance in ancient Egyptian civilization.
    """
                   }
class_Location = {
    'Akhenaten': "https://maps.app.goo.gl/CQ1yY1d9a22LxjKW8",
    'Bent pyramid for senefru': 'https://maps.app.goo.gl/F9iKKWojxRQWKjmx7',
    'Colossal Statue of Ramesses II': 'https://maps.app.goo.gl/9XGHMtPu1jkFbTRL8', 
    'Colossoi of Memnon': 'https://maps.app.goo.gl/Ts5WCBrY8ptAsdra7',
    'Goddess Isis with her child': 'No Location',
    'Hatshepsut': 'https://maps.app.goo.gl/gBYFY1134PCskwvz5',
    'Khafre Pyramid': 'https://maps.app.goo.gl/mkZTgLBCSaJJoBqt9',
    'King Thutmose III': 'https://maps.app.goo.gl/njb73LNxDUUBQgWv5',
    'Mask of Tutankhamun': 'https://maps.app.goo.gl/8qUrwuAEHSytts1g6', 
    'Nefertiti': 'https://maps.app.goo.gl/LMhwYxzsKzJm7iNn7', 
    'Pyramid_of_Djoser': 'https://maps.app.goo.gl/32hYwCzepozsQLKG7', 
    'Ramessum': 'https://maps.app.goo.gl/dbJP5oP1snm8uSEk8',
    'Statue of King Zoser': 'https://maps.app.goo.gl/7c331DE1ozzM3HdBA', 
    'Statue of Tutankhamun with Ankhesenamun': 'https://maps.app.goo.gl/8qUrwuAEHSytts1g6',
    'Temple_of_Isis_in_Philae': 'https://maps.app.goo.gl/GDNr5ENYmwQ75tRc7', 
    'Temple_of_Kom_Ombo': 'https://maps.app.goo.gl/Yjjck1zVeTvrhNJ67', 
    'The Great Temple of Ramesses II': 'Thttps://maps.app.goo.gl/N7tN4QtrcVe2hez19',
    'amenhotep iii and tiye': 'https://maps.app.goo.gl/PnSoqxLH1eQiQTdu6',
    'bust of ramesses ii': 'https://maps.app.goo.gl/78p5MCx6LbqGbEAX6',
    'head Statue of Amenhotep iii': 'https://maps.app.goo.gl/PnSoqxLH1eQiQTdu6', 
    'menkaure pyramid': 'https://maps.app.goo.gl/nurHiRxuXknGoVrJ8',
    'sphinx': 'https://maps.app.goo.gl/aeFcmD8FARFR4HEs9'}
model=load_model("mobilenet_model.h5")

@app.route('/')
def index():

    return render_template('index.html', appName="Intel Image Classification")


@app.route('/predictApi', methods=["POST"])
def api():
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        image_arr = preprossing(image)
        print("Model predicting ...")
        result = model.predict(image_arr)
        print("Model predicted")
        ind = np.argmax(result)
        prediction_class = classes[ind]
        prediction_text = class_texts[prediction_class]  # Get the text for the predicted class
        prediction_Location = class_Location[prediction_class]
        print(prediction_class)
        print(prediction_text)
        print(prediction_Location)
        return jsonify({'prediction': prediction_class, 'prediction_text': prediction_text ,'prediction_Location':prediction_Location})
    except:
        return jsonify({'Error': 'Error occur'})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        image = request.files['fileup']
        print("image loaded....")
        image_arr= preprossing(image)
        print("predicting ...")
        result = model.predict(image_arr)
        print("predicted ...")
        ind = np.argmax(result)
        prediction_class = classes[ind]
        prediction_text = class_texts[prediction_class]  # Get the text for the predicted class
        prediction_Location = class_Location[prediction_class]
        print(prediction_class)
        print(prediction_text)
        print(prediction_Location)
        return render_template('index.html', prediction=prediction_class, prediction_text=prediction_text,prediction_Location=prediction_Location, image='static/IMG/', appName="Intel Image Classification")
    else:
        return render_template('index.html',appName="Intel Image Classification")



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000,debug=True)
