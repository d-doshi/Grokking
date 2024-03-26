def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--link", type=str, help="insert the link to Github file")

    args = parser.parse_args()

    print(args.link.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/"))
    

if __name__ == "__main__":
    main()