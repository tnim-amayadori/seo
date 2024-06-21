from config import secrets
from datetime import timedelta
from generative import anticipate_cost, cluster, common_util, get_intent, vectorize
import logging
import time

_path_data = 'data'
_input_path = 'data/input_sample.csv'


def _main(input_path=_input_path):
    # Initialize.
    common_util.initialize_logging(_path_data)

    try:
        logging.info("Starting process.")

        secrets.set_api_key()

        # API利用料を予測
        out_cost = common_util.get_daily_path(get_intent.out_cost)
        total_usd, total_jpy = get_intent.pre_anticipate(input_path, out_cost)

        out_cost = common_util.get_daily_path(vectorize.out_cost)
        tmp_usd, tmp_jpy = vectorize.pre_anticipate(input_path, out_cost)
        total_usd += tmp_usd
        total_jpy += tmp_jpy

        out_cost = common_util.get_daily_path(cluster.out_cost_pre)
        tmp_usd, tmp_jpy = cluster.pre_anticipate(input_path, out_cost)
        total_usd += tmp_usd
        total_jpy += tmp_jpy

        msg = anticipate_cost.print_cost(total_usd, total_jpy, pre_msg="Total API Cost [pre]")

        # 処理実行の確認
        user_input = input(f"{msg} \nDo you want to proceed? (y/n) [default is n=No]: ").strip().lower()
        if user_input != 'y':
            logging.info("Process aborted by user")
            return

        # 各モジュールの処理実行
        start_time = time.time()

        elapsed_time = time.time() - start_time
        formatted_time = str(timedelta(seconds=elapsed_time))
        logging.info(f"module1 executed in {formatted_time} seconds.")

        logging.info("All modules executed successfully.")

    except Exception as e:
        logging.exception(f"An error occurred: {e}")

    finally:
        logging.info("Process completed.")


if __name__ == "__main__":
    _main()
