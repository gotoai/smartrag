from smartrag.cli.main import main

# The guard matters: a sync's embedding workers are started with `spawn`, which imports
# this module again in each worker under another name.
if __name__ == "__main__":
    main()
