#include <future>
#include <cstddef>
#include <vector>
#include <queue>
#include <mutex>
#include <thread>
#include <functional>
#include <type_traits>
#include <condition_variable>

class ThreadPool {
    private:
        std::vector<std::thread> workers;
        std::queue<std::function<void()>> tasks;
        std::mutex queueMutex;
        std::condition_variable condition;

        bool stop;
    public:
        ThreadPool(size_t threadsCount);
        template<class F, class... Args>

        auto enqueue(F&& f, Args&&... args)
            -> std::future<std::invoke_result_t<F, Args...>>
        {
            using return_type = std::invoke_result_t<F, Args...>;

            auto task = std::make_shared<std::packaged_task<return_type()>>(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...)
            );

            std::future<return_type> res = task->get_future();

            {
                std::unique_lock<std::mutex> lock(queueMutex);

                if (stop) {
                    throw std::runtime_error("Trying to add task to closed Pool");
                }

                tasks.emplace(
                    [task]() {
                        (*task)();
                    }
                );
            }

            condition.notify_one();

            return res;
        }

        ~ThreadPool();
};