import argparse

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="CS410 Recommendation System"
    )

    # Add arguments
    parser.add_argument(
        "-b", "--book",
        type=str,
        required=True,
        help="Book title to get recommendations for"
    )
    parser.add_argument(
        "-k", "--top_k",
        type=int,
        required=False,
        default=10, 
        help="How many recommendations to return"
    )

    # Parse the arguments
    args = parser.parse_args()

    print(args.book)
    print(args.top_k)

if __name__ == "__main__":
    main()
