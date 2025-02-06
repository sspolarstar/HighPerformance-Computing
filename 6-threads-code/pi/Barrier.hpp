#include <condition_variable>
#include <mutex>

/**
 * @brief A thread syncronization object
 * Initialize the barrier with the number of threads that will be synchronized.
 * Each thread calls wait and blocks until all threads are waiting.
 * The barrier can be reused to synchronize the _same_ number of threads.
 * 
 */
class Barrier
{
public:
  /**
   * @brief Construct a new Barrier object
   * 
   * @param n number of threads to be synchronized
   */
  Barrier(const int n): n(n), count(n) {} 

  /**
   * @brief blocks until all threads are waiting
   * 
   */
  void wait()
  {
    std::unique_lock<std::mutex> lk(m);
    --count;
    if (count != 0)
    {
      cv.wait(lk); 
    }
    else
    {
      cv.notify_all(); 
      count = n;
    }
  }

private:
  const int n;
  int count;
  std::condition_variable cv;
  std::mutex m;
};
