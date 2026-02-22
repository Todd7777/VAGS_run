import os
import torch
import numpy as np
import csv
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image
from transformers import CLIPProcessor, CLIPModel
import lpips
from torchvision import transforms

# ==================== CONFIGURATION ====================
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16

# Dossier contenant les images sources (ex: Data/Images/bear.png)
INPUT_ROOT = "Data/Images" 
OUTPUT_DIR = "outputs/FINAL_DUEL_BASELINE_VS_WANG"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== DATASET NETTOYÉ & COMPLET ====================
DATASET = [
    {
        "filename": "bear.png",
        "source": "A large brown bear walking through a stream of water. The stream appears to be shallow, as the bear is able to walk through it without any difficulty.",
        "targets": [
            ("A large black bear walking through a stream of water. The stream appears to be shallow, as the black bear is able to walk through it without any difficulty.", "black_bear"),
            ("A large panda bear walking through a stream of water. The stream appears to be shallow, as the panda bear is able to walk through it without any difficulty.", "panda_bear"),
            ("A large polar bear walking through a stream of water. The stream appears to be shallow, as the polar bear is able to walk through it without any difficulty.", "polar_bear"),
            ("A large sculpture of a brown bear walking through a stream of water. The stream appears to be shallow, as the sculpture of the bear is able to walk through it without any difficulty.", "sculpture")
        ]
    },
    {
        "filename": "bear_grass.png",
        "source": "A large brown bear walking through a grassy field.",
        "targets": [
            ("A large black bear walking through a grassy field.", "black_bear"),
            ("A large panda bear walking through a grassy field.", "panda_bear"),
            ("A large polar bear walking through a grassy field.", "polar_bear"),
            ("A large sculpture of a brown bear walking through a grassy field.", "sculpture"),
            ("A large lion walking through a grassy field.", "lion"),
            ("A large tiger walking through a grassy field.", "tiger")
        ]
    },
    {
        "filename": "beer_glass.png",
        "source": "A wine glass filled with beer.",
        "targets": [("A wine glass filled with a vibrant red cocktail, garnished with an orange wedge.", "cocktail")]
    },
    {
        "filename": "bikes.png",
        "source": "A bicycle parked on the sidewalk in front of a red brick building. The bicycle is positioned close to the door of the building, making it easily accessible for the owner. The building has a red brick exterior, giving it a distinctive appearance.",
        "targets": [
            ("A motorcycle parked on the sidewalk in front of a red brick building.", "motorcycle"),
            ("A scooter parked on the sidewalk in front of a red brick building.", "scooter"),
            ("A harley davidson motorcycle parked on the sidewalk in front of a red brick building.", "harley"),
            ("A vespa scooter parked on the sidewalk in front of a red brick building.", "vespa"),
            ("A green bicycle parked on the sidewalk in front of a red brick building.", "green_bicycle"),
            ("A yellow bicycle parked on the sidewalk in front of a red brick building.", "yellow_bicycle")
        ]
    },
    {
        "filename": "boat_silhouette.png",
        "source": "A serene scene of a lake with a silhouette of a sailboat floating on the water. The silhouette of the boat is positioned in the middle of the lake, slightly to the left. The sun is setting in the background, casting a warm glow over the scene.",
        "targets": [
            ("A serene scene of a lake with a sailboat floating on the water. The sailboat has white sails and red hull.", "white_sails"),
            ("A serene scene of a lake with a silhouette of a paper boat floating on the water.", "paper_boat")
        ]
    },
    {
        "filename": "brown_owl.png",
        "source": "A small, brown owl standing on a patch of grass.",
        "targets": [
            ("A small, white owl standing on a patch of grass.", "white_owl"),
            ("A small glass sculpture of a brown owl standing on a patch of grass.", "glass_sculpture"),
            ("A small origami owl standing on a patch of grass.", "origami_owl"),
            ("A pigeon standing on a patch of grass.", "pigeon")
        ]
    },
    {
        "filename": "bus.png",
        "source": "A yellow van parked in front of a house, outside a garage. The van is of a classic model. The house has a brown exterior, and there is a tree nearby. There is a spare tire hanged between the headlights of the van.",
        "targets": [
            ("A SUV parked in front of a house, outside a garage. The SUV is of a modern model, possibly electric.", "SUV"),
            ("A yellow van parked in front of a house, outside a garage. There is a big Volkswagen logo on the front of the van.", "vw_logo"),
            ("A pink van parked in front of a house, outside a garage.", "pink_van"),
            ("A military jeep parked in front of a house, outside a garage.", "military_jeep")
        ]
    },
    {
        "filename": "butterflies.png",
        "source": "A vibrant orange flower with two black and white butterflies perched on it.",
        "targets": [("A vibrant orange flower with two yellow butterflies perched on it.", "yellow_butterflies")]
    },
    {
        "filename": "butterfly.png",
        "source": "A beautiful purple flower with a white and black butterfly perched on top of it.",
        "targets": [
            ("A beautiful purple flower with an orange butterfly perched on top of it.", "orange_butterfly"),
            ("A beautiful red flower with a white and black butterfly perched on top of it.", "red_flower"),
            ("A beautiful purple flower with a hummingbird perched on top of it.", "hummingbird")
        ]
    },
    {
        "filename": "cake.png",
        "source": "A three-layer cake with white frosting, placed on a wooden table. The cake is adorned with a variety of fruits. The cake is presented on a white plate.",
        "targets": [
            ("A three-layer cake with white frosting, placed on a wooden table. The cake is adorned with a variety of berries.", "berries"),
            ("A three-layer cake with chocolate frosting, placed on a wooden table. The cake is adorned with a variety of berries.", "chocolate_berries"),
            ("A three-layer wedding cake with white frosting, placed on a wooden table.", "wedding_cake"),
            ("A three-layer wedding cake with white frosting, placed on a wooden table. The cake is adorned with a lot of strawberries.", "strawberries")
        ]
    },
    {
        "filename": "cake_red_blueberries.png",
        "source": "A slice of red velvet cake with white frosting, topped with blueberries.",
        "targets": [
            ("A slice of chocolate cake with white frosting, topped with blueberries.", "chocolate"),
            ("A slice of red velvet cake with white frosting, topped with raspberries.", "raspberries")
        ]
    },
    {
        "filename": "castle.png",
        "source": "A majestic castle with a tall, pointed roof, situated on a hillside.",
        "targets": [("A majestic castle made out of lego bricks with a tall, pointed roof, situated on a hillside.", "lego_castle")]
    },
    {
        "filename": "cat_and_dog.png",
        "source": "A dog and a cat sitting together on a sidewalk.",
        "targets": [
            ("A dog and a cat made out of lego sitting together on a sidewalk.", "lego"),
            ("A bronze sculpture of a dog and a cat sitting together on a sidewalk.", "bronze")
        ]
    },
    {
        "filename": "cat_and_dog2.png",
        "source": "A gray cat and a brown dog on a floor, some distance between them.",
        "targets": [
            ("A tiger and a brown dog on a floor.", "tiger_dog"),
            ("A raccoon and a brown dog on a floor.", "raccoon_dog"),
            ("A koala and a brown dog on a floor.", "koala_dog"),
            ("A red panda and a brown dog on a floor.", "red_panda_dog"),
            ("A tiger and a brown bear on a floor.", "tiger_bear"),
            ("A lion and a brown bear on a floor.", "lion_bear")
        ]
    },
    {
        "filename": "cat_crown.png",
        "source": "A gray cat sitting on a black cloth, wearing a crown.",
        "targets": [
            ("A gray cat sitting on a black cloth, wearing a top hat.", "black_top_hat"),
            ("A lioness sitting on a black cloth, wearing a top hat.", "lioness_top_hat"),
            ("A rabbit sitting on a black cloth, wearing a top hat.", "rabbit_top_hat")
        ]
    },
    {
        "filename": "cat.png",
        "source": "A small, fluffy kitten sitting in a grassy field.",
        "targets": [
            ("A small puppy sitting in a grassy field.", "puppy"),
            ("A small lion cub sitting in a grassy field.", "lion_cub"),
            ("A small tiger cub sitting in a grassy field.", "tiger_cub"),
            ("A small bear cub sitting in a grassy field.", "bear_cub"),
            ("A small fluffy fox sitting in a grassy field.", "fox"),
            ("A small, fluffy poodle puppy sitting in a grassy field.", "poodle_puppy"),
            ("A small wooden sculpture of a kitten sitting in a grassy field.", "wooden_sculpture")
        ]
    },
    {
        "filename": "cat_dog_car.png",
        "source": "A blue-gray Audi car parked in a grassy area. A white dog sitting on the grass, next to the car. A cat laying on the hood of the car.",
        "targets": [
            ("A blue-gray Audi car parked in a grassy area. A Husky dog sitting on the grass, next to the car. A tiger cab laying on the hood of the car.", "husky_tiger"),
            ("A blue-gray Audi car parked in a grassy area. A Husky dog sitting on the grass, next to the car. A lion cab laying on the hood of the car.", "husky_lion"),
            ("A blue-gray Audi car parked in a grassy area. A gray wolf sitting on the grass, next to the car. A tiger cab laying on the hood of the car.", "wolf_tiger"),
            ("A blue-gray Audi car parked in a grassy area. A gray wolf sitting on the grass, next to the car. A lion cab laying on the hood of the car.", "wolf_lion"),
            ("A blue-gray Audi car parked in a grassy area. A Dalmatian dog sitting on the grass, next to the car. A tiger cab laying on the hood of the car.", "dalmatian_tiger"),
            ("A blue-gray Audi car parked in a grassy area. A Dalmatian dog sitting on the grass, next to the car. A lion cab laying on the hood of the car.", "dalmatian_lion"),
            ("A blue-gray Audi car parked in a grassy area. A Dalmatian dog sitting on the grass, next to the car.", "dalmatian_no_cat"),
            ("A blue-gray Audi car parked in a grassy area. A Husky dog sitting on the grass, next to the car.", "husky_no_cat"),
            ("A blue-gray Audi car parked in a grassy area. A wolf sitting on the grass, next to the car.", "wolf_no_cat")
        ]
    },
    {
        "filename": "cat_ginger.png",
        "source": "An orange and white cat sitting.",
        "targets": [
            ("A silver cat sculpture.", "silver"),
            ("A tiger sitting.", "tiger")
        ]
    },
    {
        "filename": "cat_stone.png",
        "source": "A small gray kitten sitting on a stone ledge.",
        "targets": [
            ("A small black kitten sitting on a stone ledge.", "black_kitten"),
            ("A small puppy sitting on a stone ledge.", "puppy"),
            ("A small sculpture of a kitten sitting on a stone ledge.", "sculpture"),
            ("A small tiger sitting on a stone ledge.", "tiger"),
            ("A small lion sitting on a stone ledge.", "lion"),
            ("A small pig sitting on a stone ledge.", "pig")
        ]
    },
    {
        "filename": "clown_fish.png",
        "source": "A vibrant underwater scene with a clownfish swimming among various coral reefs.",
        "targets": [
            ("A vibrant underwater scene with a goldfish swimming among various coral reefs.", "goldfish"),
            ("A vibrant underwater scene with a small sea turtle swimming among various coral reefs.", "sea_turtle"),
            ("A vibrant underwater scene with a shark swimming among various coral reefs.", "shark"),
            ("A vibrant underwater scene with a seahorse swimming among various coral reefs.", "seahorse")
        ]
    },
    {
        "filename": "coconut_out.png",
        "source": "The image features a coconut shell filled with water, creating a unique and artistic scene.",
        "targets": [
            ("The image features a human head filled with water.", "head"),
            ("The image features a baseball filled with water.", "baseball"),
            ("The image features a giant cup filled with water.", "cup")
        ]
    },
    {
        "filename": "corgi.png",
        "source": "A brown and white dog sitting on a dirt ground near a body of water.",
        "targets": [
            ("A brown and white dog made out of lego bricks sitting on a dirt ground.", "lego_bricks"),
            ("A wooden sculpture of a brown and white dog made sitting on a dirt ground.", "wooden_sculpture"),
            ("A red fox sitting on a dirt ground near a body of water.", "red_fox")
        ]
    },
    {
        "filename": "cupcake.png",
        "source": "A small chocolate cupcake with light brown whipped cream on top.",
        "targets": [
            ("A small vanilla cupcake with whipped cream on top.", "vanilla"),
            ("A small red velvet cupcake with whipped cream on top.", "red_velvet")
        ]
    },
    {
        "filename": "dalmatian.png",
        "source": "A Dalmatian dog sitting on a white background.",
        "targets": [
            ("A Dalmatian dog sitting on a white background. The dog is looking to the camera.", "looking_camera"),
            ("A cheetah sitting on a white background.", "cheetah_looking_camera"),
            ("A crochet Dalmatian dog sitting on a white background.", "crochet_dog_looking_camera")
        ]
    },
    {
        "filename": "deer_silhouette.png",
        "source": "A silhouette of a deer standing on a grassy bank near a lake.",
        "targets": [("A silhouette of a male deer with beautiful antlers standing on a grassy bank near a lake.", "male_deer_antlers")]
    },
    {
        "filename": "dog.png",
        "source": "A small black and brown dog sitting on a lush green field.",
        "targets": [
            ("A small black and brown dog standing on a lush green field.", "standing"),
            ("A small black and brown dog puppet sitting on a lush green field.", "puppet"),
            ("A small black and brown crochet dog sitting on a lush green field.", "crochet"),
            ("A small black and brown sculpture of a dog sitting on a lush green field.", "sculpture"),
            ("A small black and brown dog made out of lego bricks sitting on a lush green field.", "lego_bricks"),
            ("A small black and brown dog sitting on a lush green field. The dog is wearing a red top hat.", "red_top_hat"),
            ("A small black and brown dog sitting on a lush green field. The dog is wearing a jeweled crown.", "jeweled_crown"),
            ("A small black and brown poodle sitting on a lush green field.", "poodle")
        ]
    },
    {
        "filename": "dog_snow.png",
        "source": "A German Shepherd dog standing in a snowy field.",
        "targets": [
            ("A wooden sculpture of a dog standing in a snowy field.", "wooden_sculpture"),
            ("A wolf standing in a snowy field.", "wolf"),
            ("A fox standing in a snowy field.", "fox"),
            ("A Husky dog standing in a snowy field.", "husky"),
            ("A Golden Retriever dog standing in a snowy field.", "golden"),
            ("A bear standing in a snowy field.", "bear"),
            ("A baby deer standing in a snowy field.", "baby_deer")
        ]
    },
    {
        "filename": "dogs.png",
        "source": "Two dingo dogs standing close to each other in a wooded area.",
        "targets": [
            ("Two red foxes standing close to each other in a wooded area.", "red_foxes"),
            ("Two wolves standing close to each other in a wooded area.", "wolves"),
            ("Two Huskies standing close to each other in a wooded area.", "huskies")
        ]
    },
    {
        "filename": "dogs2.png",
        "source": "Two dogs standing on a grassy hillside. One dog positioned to the right of the other.",
        "targets": [("Two wolves standing on a grassy hillside.", "wolf")]
    },
    {
        "filename": "drawing_horse.png",
        "source": "A black and white sketch of a horse.",
        "targets": [("A black and white sketch of a unicorn.", "unicorn")]
    },
    {
        "filename": "duck.png",
        "source": "A brown duck swimming in a lake. The scene is set against a black background.",
        "targets": [
            ("A colorful duck swimming in a lake.", "colorful_duck"),
            ("A black swan swimming in a lake.", "black_swan"),
            ("A white swan swimming in a lake.", "white_swan"),
            ("A goose swimming in a lake.", "goose")
        ]
    },
    {
        "filename": "flowers_draw.png",
        "source": "A painting of three red flowers, each with a long stem.",
        "targets": [
            ("A painting of three sunflowers, each with a long stem.", "sunflowers"),
            ("A painting of three tulips, each with a long stem.", "tulips")
        ]
    },
    {
        "filename": "flowers.png",
        "source": "A vase filled with a beautiful bouquet of pink, red and white flowers.",
        "targets": [
            ("A vase filled with a beautiful bouquet of orange, yellow and white flowers.", "orange_yellow_white"),
            ("A vase filled with a beautiful bouquet of blue, purple and white flowers.", "blue_purple_white")
        ]
    },
    {
        "filename": "free_wifi.png",
        "source": "A wooden table with a black board on it. The board displays the words 'FREE WIFI' in white letters.",
        "targets": [
            ("A wooden table with a black board on it. The board displays the words 'FREE BEER' in white letters.", "free_beer"),
            ("A wooden table with a black board on it. The board displays the words 'FREE HUGS' in white letters.", "free_hugs")
        ]
    },
    {
        "filename": "gas_station.png",
        "source": "A gas station with a white and red sign that reads 'CAFE'.",
        "targets": [
            ("A gas station with a white and red sign that reads 'CVPR'.", "cvpr"),
            ("A gas station with a white and red sign that reads 'ICCV'.", "iccv"),
            ("A gas station with a white and red sign that reads 'ECCV'.", "eccv"),
            ("A gas station with a white and red sign that reads 'FOOD'.", "food"),
            ("A gas station with a white and red sign that reads 'LOVE'.", "love"),
            ("A gas station with a white and red sign that reads 'FREE'.", "free")
        ]
    },
    {
        "filename": "geese.png",
        "source": "A flock of geese flying together in the sky.",
        "targets": [
            ("A flock of ducks flying together in the sky.", "ducks"),
            ("A flock of swans flying together in the sky.", "swans"),
            ("A flock of flamingos flying together in the sky.", "flamingos")
        ]
    },
    {
        "filename": "gray_bird.png",
        "source": "A small blue and gray bird perched on a wooden fence or post.",
        "targets": [
            ("A small blue and gray origami bird perched on a wooden fence or post.", "origami_bird"),
            ("A small red and gray bird perched on a wooden fence or post.", "red_bird"),
            ("A small sculpture of a blue and gray bird perched on a wooden fence or post.", "sculpture"),
            ("A small golden sculpture of a bird perched on a wooden fence or post.", "golden_sculpture")
        ]
    },
    {
        "filename": "groceries.png",
        "source": "A grocery list writing on a piece of brown paper with a black marker, which includes BREAD, EGGS, and MILK.",
        "targets": [
            ("A grocery list writing on a piece of brown paper which includes BACON, BREAD, EGGS, and MILK.", "bacon"),
            ("A grocery list writing on a piece of brown paper which includes BREAD, COFFEE, and MILK.", "coffee"),
            ("A grocery list writing on a piece of brown paper which includes CVPR, EGGS, and MILK.", "cvpr"),
            ("A grocery list writing on a piece of brown paper which includes ECCV, EGGS, and MILK.", "eccv"),
            ("A grocery list writing on a piece of brown paper which includes ICCV, EGGS, and MILK.", "iccv")
        ]
    },
    {
        "filename": "horse_silhouette.png",
        "source": "A black silhouette of a horse, showcasing its elegant and powerful form.",
        "targets": [("A black silhouette of a unicorn, showcasing its elegant and powerful form.", "unicorn")]
    },
    {
        "filename": "horse.png",
        "source": "A white horse running through a grassy field.",
        "targets": [
            ("A white unicorn running through a grassy field.", "unicorn"),
            ("A pink toy horse running through a grassy field.", "pink_toy_horse"),
            ("A sculpture bronze horse running through a grassy field.", "bronze_sculpture"),
            ("A brown horse running through a grassy field.", "brown_horse")
        ]
    },
    {
        "filename": "iguana.png",
        "source": "A large orange lizard sitting on a rock near the ocean.",
        "targets": [
            ("A large green lizard sitting on a rock near the ocean.", "green_lizard"),
            ("A large blue lizard sitting on a rock near the ocean. The blue lizard is wearing a top hat.", "blue_lizard_top_hat"),
            ("A large frog sitting on a rock near the ocean.", "frog"),
            ("A large crocodile sitting on a rock near the ocean.", "crocodile"),
            ("A large dragon sitting on a rock near the ocean.", "dragon")
        ]
    },
    {
        "filename": "japanese_castle.png",
        "source": "A large, white, Japanese castle with a distinctive pagoda-style roof.",
        "targets": [("A large, white, Japanese castle made out of lego bricks with a distinctive pagoda-style roof.", "lego_castle")]
    },
    {
        "filename": "jump_kick.png",
        "source": "A woman in a white uniform, kicking in the air.",
        "targets": [
            ("A shiny silver humanoid robot kicking in the air.", "robot"),
            ("A bronze statue of a woman kicking in the air.", "bronze_statue"),
            ("A golden sculpture of a woman kicking in the air.", "golden_statue"),
            ("A marble statue of a woman kicking in the air.", "marble_statue")
        ]
    },
    {
        "filename": "jump_yay.png",
        "source": "A man wearing a white shirt and black shorts, jumping in the air with his left arm raised.",
        "targets": [
            ("A shiny silver humanoid robot jumping in the air.", "robot"),
            ("A bronze statue of a man jumping in the air.", "bronze_statue"),
            ("A golden sculpture of a man jumping in the air.", "golden_statue"),
            ("Superman jumping in the air.", "superman")
        ]
    },
    {
        "filename": "kid_running.png",
        "source": "A young boy running through a grassy field.",
        "targets": [
            ("A shiny silver robot running through a grassy field.", "robot"),
            ("A futuristic robot running through a grassy field.", "futuristic_robot"),
            ("A sculpture of a young boy running through a grassy field.", "sculpture"),
            ("A wooden sculpture of a young boy running through a grassy field.", "wooden_sculpture")
        ]
    },
    {
        "filename": "lighthouse.png",
        "source": "The image features a tall white lighthouse standing prominently on a hill.",
        "targets": [
            ("The image features a space rocket prominently on a hill.", "space_rocket"),
            ("The image features Rapunzel's tower standing prominently on a hill.", "rapunzel_tower"),
            ("The image features the Eiffel tower standing prominently on a hill.", "eiffel_tower"),
            ("The image features a tall obelisk standing prominently on a hill.", "obelisk"),
            ("The image features Big ben clock tower standing prominently on a hill.", "big_ben")
        ]
    },
    {
        "filename": "luna.png",
        "source": "A neon sign for a restaurant called Luna.",
        "targets": [
            ("A neon sign for a restaurant called Sol.", "sol"),
            ("A neon sign for a restaurant called CVPR.", "cvpr"),
            ("A neon sign for a restaurant called ICCV.", "iccv"),
            ("A neon sign for a restaurant called ECCV.", "eccv"),
            ("A neon sign for a restaurant with the word WELCOME.", "welcome"),
            ("A neon sign for a restaurant with the word HI.", "hi"),
            ("A neon sign for a restaurant with an heart on it.", "heart")
        ]
    },
    {
        "filename": "meditation.png",
        "source": "A woman sitting on the floor in a room and meditating.",
        "targets": [("A wooden statue of a woman sitting on the floor in a room and meditating.", "wooden_statue")]
    },
    {
        "filename": "meditation1.png",
        "source": "A man sitting on the ground and appears to be in a relaxed position, meditating.",
        "targets": [
            ("A sand sculpture of man sitting on the ground.", "sand_sculpture"),
            ("A golden sculpture of buddha sitting on the ground.", "golden_buddha_statue"),
            ("A wooden statue of a buddha sitting on the ground.", "wooden_buddha_statue"),
            ("A golden sculpture of man sitting on the ground.", "golden_statue"),
            ("A wooden sculpture of man sitting on the ground.", "wooden_statue")
        ]
    },
    {
        "filename": "milk.png",
        "source": "A glass of milk placed on a wooden table.",
        "targets": [
            ("A glass of chocolate milk placed on a wooden table.", "chocolate_milk"),
            ("A glass of beer placed on a wooden table.", "beer"),
            ("A glass of milkshake placed on a wooden table.", "milkshake"),
            ("A glass of milk placed on a wooden table. There is whipped cream on top of the milk.", "whipped_cream")
        ]
    },
    {
        "filename": "mountain_black.png",
        "source": "A large, green, hill in the middle of a vast, black, and barren landscape.",
        "targets": [
            ("A large egyptian pyramid in the middle of a vast, black, and barren landscape.", "egyptian_pyramid"),
            ("A large mesoamerican pyramid in the middle of a vast, black, and barren landscape.", "mesoamerican_pyramid"),
            ("A large, elegant wedding cake with multiple tiers in the middle of a vast, black, and barren landscape.", "wedding_cake"),
            ("A kugelhopf cake powdered with sugar in the middle of a vast, black, and barren landscape.", "kugelhopf"),
            ("A large volcano, during a violent eruption, in the middle of a vast, black, and barren landscape.", "volcano")
        ]
    },
    {
        "filename": "muffins.png",
        "source": "A plate of white muffins, decorated with a few raspberries.",
        "targets": [("A plate of white muffins, decorated with a few strawberries.", "strawberries")]
    },
    {
        "filename": "parrot.png",
        "source": "A colorful parrot perched on a tree branch.",
        "targets": [
            ("A glass sculpture of a colorful parrot perched on a tree branch.", "glass_sculpture"),
            ("An origami of a colorful parrot perched on a tree branch.", "origami"),
            ("A gray pigeon perched on a tree branch.", "gray_pigeon"),
            ("A colorful lego parrot perched on a tree branch.", "lego_parrot")
        ]
    },
    {
        "filename": "parrot2.png",
        "source": "A colorful parrot perched on top of a tall, brightly colored flower.",
        "targets": [("A glass sculpture of a colorful parrot perched on top of a tall, brightly colored flower.", "glass_sculpture")]
    },
    {
        "filename": "parrots.png",
        "source": "Two colorful parrots sitting on a wooden stump or log.",
        "targets": [
            ("Two colorful parrots sitting on a wooden stump or log. Both parrots have a top hat on their head.", "top_hat"),
            ("Two colorful origami parrots sitting on a wooden stump or log.", "origami"),
            ("A sculpture of two colorful parrots sitting on a wooden stump or log.", "sculpture")
        ]
    },
    {
        "filename": "parrots2.png",
        "source": "Two colorful parrots perched on a rocky surface.",
        "targets": [("Two colorful parrots perched on a rocky surface. Both parrots have a crown on their head.", "crown")]
    },
    {
        "filename": "penguins.png",
        "source": "Two penguins standing in a lake.",
        "targets": [
            ("Two origami penguins standing in a lake.", "origami"),
            ("A sculpture of two penguins standing in a lake.", "sculpture")
        ]
    },
    {
        "filename": "piece_of_cake.png",
        "source": "A delicious chocolate cake with chocolate frosting, sitting on a dining table.",
        "targets": [
            ("A delicious chocolate cake with chocolate frosting and a cherry on top.", "cherry_on_top"),
            ("A delicious red velvet cake with white frosting.", "red_velvet_cake"),
            ("A delicious matcha cake with green frosting.", "matcha_cake")
        ]
    },
    {
        "filename": "pizza.png",
        "source": "A large, cheesy pizza sitting on a wooden pizza board.",
        "targets": [("A large, cheesy pizza, topped with pineapple and ham.", "pineapple_ham")]
    },
    {
        "filename": "pizza_board.png",
        "source": "A large pizza with cheese and bits of meat.",
        "targets": [
            ("A large pizza with cheese and mushrooms.", "mushrooms"),
            ("A large pizza with cheese and sausages.", "sausage"),
            ("A large pizza with cheese, pineapple and ham.", "pineapple_ham")
        ]
    },
    {
        "filename": "pizza_slice.png",
        "source": "A pizza with thick crust, topped with graded cheese and basil.",
        "targets": [
            ("A pizza with thick crust, topped with graded cheese, basil and pepperoni.", "pepperoni"),
            ("A pizza with thick crust, topped with graded cheese, basil and mushrooms.", "mushrooms")
        ]
    },
    {
        "filename": "pizza_tomato_olive.png",
        "source": "A large pizza, topped with cheese, black olive, sliced tomatoes.",
        "targets": [
            ("A large pizza, topped with cheese, black olive, sliced tomatoes and a lot of pepperoni.", "pepperoni"),
            ("A large pizza, topped with cheese, black olive, sliced tomatoes and mushrooms.", "mushrooms")
        ]
    },
    {
        "filename": "puppies.png",
        "source": "Two adorable golden retriever puppies sitting in a grassy field.",
        "targets": [
            ("Two adorable husky puppies sitting in a grassy field.", "husky_puppies"),
            ("Two adorable bear cubs sitting in a grassy field.", "bear_cubs"),
            ("Two adorable puppets of golden retriever puppies sitting in a grassy field.", "puppets"),
            ("Two adorable golden retriever puppies laying in a grassy field.", "laying")
        ]
    },
    {
        "filename": "rabbit.png",
        "source": "A small brown and white rabbit sitting in a grassy field.",
        "targets": [
            ("A small sculpture of a rabbit sitting in a grassy field.", "sculpture"),
            ("A small puppy sitting in a grassy field.", "puppy"),
            ("A small kitten sitting in a grassy field.", "kitten")
        ]
    },
    {
        "filename": "rocks.png",
        "source": "A stack of gray rocks, laying on top of each other on a brown and gray beach.",
        "targets": [
            ("A stack of colorful macarons, laying on top of each other on a brown and gray beach.", "macarons"),
            ("A stack of oreo cookies, laying on top of each other on a brown and gray beach.", "oreo_cookies"),
            ("A tower on a brown and gray beach.", "tower"),
            ("A bonsai tree on a brown and gray beach.", "bonsai_tree"),
            ("A jenga tower game on a brown and gray beach.", "jenga_tower"),
            ("A stack of colorful wooden blocks, laying on top of each other on a brown and gray beach.", "colorful_wooden_blocks")
        ]
    },
    {
        "filename": "rooster.png",
        "source": "A large, colorful rooster standing on a tree branch.",
        "targets": [
            ("A large, colorful glass sculpture of a rooster standing on a tree branch.", "glass_sculpture"),
            ("A large, colorful origami rooster standing on a tree branch.", "origami")
        ]
    },
    {
        "filename": "sign.png",
        "source": "A large white billboard with a bold message written in black letters. The message reads, 'LOVE IS ALL YOU NEED.'",
        "targets": [
            ("A large white billboard with a bold message written in black letters. The message reads, 'CVPR IS ALL YOU NEED.'", "cvpr"),
            ("A large white billboard with a bold message written in black letters. The message reads, 'ECCV IS ALL YOU NEED.'", "eccv"),
            ("A large white billboard with a bold message written in black letters. The message reads, 'ICCV IS ALL YOU NEED.'", "iccv"),
            ("A large white billboard with a bold message written in black letters. The message reads, 'FLOW IS ALL YOU NEED.'", "flow")
        ]
    },
    {
        "filename": "steak.png",
        "source": "A steak accompanied by a side of leaf salad.",
        "targets": [
            ("A bread roll accompanied by a side of leaf salad.", "bread_roll"),
            ("A schnitzel accompanied by a side of leaf salad.", "schnitzel"),
            ("A lobster tail accompanied by a side of leaf salad.", "lobster_tail"),
            ("A crab cake accompanied by a side of leaf salad.", "crab_cake"),
            ("A chocolate cake accompanied by a side of leaf salad.", "chocolate_cake"),
            ("A raw steak accompanied by a side of leaf salad.", "raw_steak"),
            ("A quiche accompanied by a side of leaf salad.", "quiche"),
            ("Shrimps accompanied by a side of leaf salad.", "shrimps")
        ]
    },
    {
        "filename": "steak_dinner.png",
        "source": "A delicious meal consisting of sliced steak, vegetables, and sauce.",
        "targets": [("A delicious meal consisting of sliced grilled salmon, vegetables, and sauce.", "salmon")]
    },
    {
        "filename": "stop.png",
        "source": "A stop sign prominently placed in a field of yellow flowers.",
        "targets": [
            ("A sign that says CVPR prominently placed in a field of yellow flowers.", "cvpr"),
            ("A sign that says ICCV prominently placed in a field of yellow flowers.", "iccv"),
            ("A sign that says ECCV prominently placed in a field of yellow flowers.", "eccv")
        ]
    },
    {
        "filename": "stop_arrow.png",
        "source": "A stop sign above a blue sign with arrow pointing up.",
        "targets": [
            ("A sign that says CVPR above a blue sign with arrow pointing up.", "cvpr"),
            ("A sign that says ICCV above a blue sign with arrow pointing up.", "iccv"),
            ("A sign that says ECCV above a blue sign with arrow pointing up.", "eccv"),
            ("A sign that says LOVE above a blue sign with arrow pointing up.", "love"),
            ("A sign that says HOME above a blue sign with arrow pointing up.", "home"),
            ("A sign that says BEER above a blue sign with arrow pointing up.", "beer")
        ]
    },
    {
        "filename": "stop_sticker.png",
        "source": "A STOP! sticker with a red and yellow color scheme which is placed on a white background.",
        "targets": [
            ("A CVPR! sticker with a red and yellow color scheme which is placed on a white background.", "cvpr"),
            ("A ECCV! sticker with a red and yellow color scheme which is placed on a white background.", "eccv"),
            ("A ICCV! sticker with a red and yellow color scheme which is placed on a white background.", "iccv")
        ]
    },
    {
        "filename": "this_must_be_the_place.png",
        "source": "A large, colorful wall with a neon sign that reads 'this must be the place.'",
        "targets": [
            ("A large, colorful wall with a neon sign that reads 'home must be the place.'", "home"),
            ("A large, colorful wall with a neon sign that reads 'cvpr must be the place.'", "cvpr"),
            ("A large, colorful wall with a neon sign that reads 'eccv must be the place.'", "eccv"),
            ("A large, colorful wall with a neon sign that reads 'iccv must be the place.'", "iccv")
        ]
    },
    {
        "filename": "tiger.png",
        "source": "A large tiger standing in a swamp.",
        "targets": [
            ("A large lion standing in a swamp.", "lion"),
            ("A large crochet tiger standing in a swamp.", "crochet_tiger"),
            ("A large wolf standing in a swamp.", "wolf")
        ]
    },
    {
        "filename": "tree_reflect.png",
        "source": "A lone tree standing in a field at night. The tree reflection can be seen in the water below it.",
        "targets": [
            ("A lone decorated christmas tree standing in a field at night.", "christmas_tree"),
            ("A lone cherry blossom tree standing in a field at night.", "cherry_blossom_tree"),
            ("A lone tree standing in a field at night with a tent next to it.", "tent")
        ]
    },
    {
        "filename": "wolf_silhouette.png",
        "source": "A silhouette of a wolf standing on a rocky cliff, looking up at the moon.",
        "targets": [
            ("A silhouette of a robot wolf standing on a rocky cliff, looking up at the moon.", "robot_wolf"),
            ("A Husky dog standing on a rocky cliff.", "husky_dog_looking"),
            ("A silhouette of a wolf standing on a rocky cliff. The wolf appears to be looking.", "wolf_looking")
        ]
    },
    {
        "filename": "yellow_bulldog.png",
        "source": "A small, yellow, shiny dog figurine positioned in the center of the scene.",
        "targets": [
            ("A small, yellow, lion figurine.", "lion"),
            ("A small, yellow, cat figurine.", "cat"),
            ("A small, yellow, bear figurine.", "bear"),
            ("A small, yellow, deer figurine.", "deer"),
            ("A small, yellow, rabbit figurine.", "rabbit"),
            ("A small, yellow, wolf figurine.", "wolf"),
            ("A small, yellow, cow figurine.", "cow"),
            ("A small, yellow, horse figurine.", "horse"),
            ("A small, yellow, unicorn figurine.", "unicorn"),
            ("A small, yellow, origami lion.", "origami_lion"),
            ("A small, yellow, origami cat.", "origami_cat"),
            ("A small, yellow, origami bear.", "origami_bear"),
            ("A small, yellow, origami deer.", "origami_deer"),
            ("A small, yellow, origami rabbit.", "origami_rabbit"),
            ("A small, yellow, origami wolf.", "origami_wolf"),
            ("A small, yellow, origami cow.", "origami_cow"),
            ("A small, yellow, origami horse.", "origami_horse")
        ]
    }
]

# ==================== EVALUATEUR ====================
class MetricEvaluator:
    def __init__(self, device):
        print("[METRICS] Loading CLIP & LPIPS...")
        self.device = device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device).eval()
        self.to_tensor = transforms.ToTensor()

    def get_clip_score(self, image, prompt):
        inputs = self.clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad(): outputs = self.clip_model(**inputs)
        return outputs.logits_per_image.item() / 100.0 

    def get_lpips_distance(self, img_source, img_generated):
        t_src = self.to_tensor(img_source).to(self.device) * 2 - 1
        t_gen = self.to_tensor(img_generated).to(self.device) * 2 - 1
        with torch.no_grad(): dist = self.lpips_loss(t_src.unsqueeze(0), t_gen.unsqueeze(0))
        return dist.item()

# ==================== MOTEUR DE COMPARAISON ====================
class ComparisonEditor:
    def __init__(self):
        print(f"[INIT] Loading SD3 Pipeline...")
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", 
            torch_dtype=DTYPE
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.to(dtype=torch.float32)

    @torch.no_grad()
    def process(self, init_image, src_prompt, tgt_prompt, strategy):
        img_t = self.pipe.image_processor.preprocess(init_image).to(DEVICE, dtype=torch.float32)
        x0_src = (self.pipe.vae.encode(img_t).latent_dist.mode() - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        x0_src = x0_src.to(dtype=DTYPE)

        pe_s, ne_s, ppe_s, npe_s = self.pipe.encode_prompt(src_prompt, src_prompt, src_prompt, device=DEVICE)
        pe_t, ne_t, ppe_t, npe_t = self.pipe.encode_prompt(tgt_prompt, tgt_prompt, tgt_prompt, device=DEVICE)
        
        # --- CONFIGURATION DU DUEL ---
        # Baseline : SD3 Standard (Neutre)
        # Wang : Optimisé (Exponential, Kappa=3.0, M=3.0)
        
        t_start = 0.85 # Commun aux deux pour comparaison équitable
        base_cfg = 7.5
        
        # Paramètres Wang Optimisés
        kappa = 3.0
        m = 3.0
            
        steps = 50
        self.pipe.scheduler.set_timesteps(steps, device=DEVICE)
        timesteps = self.pipe.scheduler.timesteps
        start_index = int((1.0 - t_start) * steps)
        
        zt = x0_src.clone()

        for i, t_tensor in enumerate(timesteps):
            if i < start_index: continue
            
            t = t_tensor.item() / 1000.0
            dt = (timesteps[i+1].item() / 1000.0 if i+1 < len(timesteps) else 0.0) - t
            noise = torch.randn_like(x0_src)
            zt_src = (1 - t) * x0_src + t * noise
            zt_tar = zt + zt_src - x0_src 

            latents_in = torch.cat([zt_src]*2 + [zt_tar]*2)
            prompt_embeds = torch.cat([ne_s, pe_s, ne_t, pe_t])
            pooled_embeds = torch.cat([npe_s, ppe_s, npe_t, ppe_t])

            noise_pred = self.pipe.transformer(
                hidden_states=latents_in, timestep=t_tensor.expand(latents_in.shape[0]), 
                encoder_hidden_states=prompt_embeds, pooled_projections=pooled_embeds, return_dict=False
            )[0]
            
            vu_s, vc_s, vu_t, vc_t = noise_pred.chunk(4)

            # --- LOGIQUE DE STRATEGIE ---
            current_lambda = base_cfg # Par défaut (Baseline)
            
            if strategy == "Wang":
                # Calcul du conflit
                v_s_vec = vc_s
                v_t_vec = vc_t
                diff = v_t_vec - v_s_vec
                norm_diff = torch.norm(diff, p=2, dim=(1,2,3)).mean().item()
                norm_src = torch.norm(v_s_vec, p=2, dim=(1,2,3)).mean().item()
                relative_conflict = norm_diff / (norm_src + 1e-5)
                
                # Formule Exponentielle
                progress = (t_start - t) / t_start
                sigma = 1 / (1 + np.exp(-12 * (progress - 0.5))) # Centré
                
                control_term = 2 * sigma - 1 
                conflict_score = np.tanh(relative_conflict / m)
                modulation = np.exp(kappa * control_term * conflict_score)
                current_lambda = base_cfg * modulation

            # Application du CFG
            v_source_final = vu_s + 3.5 * (vc_s - vu_s)
            v_target_final = vu_t + current_lambda * (vc_t - vu_t)
            
            zt = zt + dt * (v_target_final - v_source_final)

        zt_dec = (zt / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        image = self.pipe.vae.decode(zt_dec.to(torch.float32), return_dict=False)[0]
        image = torch.clamp(image, -1.0, 1.0)
        res_img = self.pipe.image_processor.postprocess(image, output_type="pil")[0]
        return res_img

if __name__ == "__main__":
    editor = ComparisonEditor()
    evaluator = MetricEvaluator(DEVICE)
    
    STRATEGIES = ["Baseline", "Wang"]
    
    print(f"[START] Grand Benchmark Final (Baseline vs Wang)...")
    
    csv_path = os.path.join(OUTPUT_DIR, "final_results.csv")
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['Image', 'Target', 'Strategy', 'CLIP', 'LPIPS', 'Filename']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in DATASET:
            # Gestion des chemins (Nettoyage automatique si 'flowedit_data' est présent ou non)
            filename_clean = item["filename"].replace("flowedit_data/", "")
            full_path = os.path.join(INPUT_ROOT, filename_clean)
            
            # Fallback si le fichier est dans un sous-dossier flowedit_data
            if not os.path.exists(full_path):
                full_path_alt = os.path.join(INPUT_ROOT, "flowedit_data", filename_clean)
                if os.path.exists(full_path_alt):
                    full_path = full_path_alt
                else:
                    print(f"[SKIP] Image introuvable : {item['filename']}")
                    continue
                
            print(f"\nProcessing {filename_clean}...")
            init_img = load_image(full_path).resize((1024,1024))
            base_name = os.path.splitext(filename_clean)[0]
            
            for tgt_prompt, code in item["targets"]:
                for strat in STRATEGIES:
                    # On ne réimprime pas tout le temps pour garder le log propre
                    if strat == "Baseline": print(f"   -> {code} [Baseline vs Wang]")
                    
                    res_img = editor.process(init_img, item["source"], tgt_prompt, strat)
                    
                    # Metrics
                    inputs = evaluator.clip_processor(text=[tgt_prompt], images=res_img, return_tensors="pt", padding=True).to(DEVICE)
                    clip_s = evaluator.clip_model(**inputs).logits_per_image.item() / 100.0
                    lpips_d = evaluator.get_lpips_distance(init_img, res_img)
                    
                    print(f"      [{strat}] CLIP: {clip_s:.4f} | LPIPS: {lpips_d:.4f}")
                    
                    fname = f"{base_name}_{code}_{strat}.jpg"
                    res_img.save(os.path.join(OUTPUT_DIR, fname))
                    
                    writer.writerow({
                        'Image': base_name,
                        'Target': code,
                        'Strategy': strat,
                        'CLIP': f"{clip_s:.4f}",
                        'LPIPS': f"{lpips_d:.4f}",
                        'Filename': fname
                    })
                    csv_file.flush()
                    torch.cuda.empty_cache()

    print(f"\n[DONE] Benchmark complet terminé dans : {OUTPUT_DIR}")