package org.jabe.neverland.io;

import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class ConCurrentTest {

	public static void main(String[] args) {
		// MyRunnable.testSync1();
		// ThreadA.testWait();
		// Producer.testProducer();
		// TestJoin.testJoin();
		// TestDaemon.testDaemon();
		// Godown.testProducer();
//		TestReentrantLock.testReentrantLock();
		TestReentrantReadWriteLock.testReentrantReadWriteLock();
	}

	/**
	 * 
	 * 测试同步关键字的使用,核心的思想是:时刻明确自己同步的是哪个一个实例对象! 当然,静态方法同步的则是Class对象.
	 * 果断避免:静态方法和实例方法相互调用,然后又是各种同步操作,JVM都帮不了你!
	 * 
	 */
	public static class MyRunnable implements Runnable {
		private static void testSync1() {
			MyRunnable r = new MyRunnable();
			Thread ta = new Thread(r, "Thread-A");
			Thread tb = new Thread(r, "Thread-B");
			ta.start();
			tb.start();
		}

		private Foo foo = new Foo();

		public void run() {
			for (int i = 0; i < 3; i++) {
				this.fix(30);
			}
		}

		/**
		 * 同步实例方法等效于手动同步这个对象的实例
		 * 
		 * @param y
		 * @return
		 */
		public synchronized int fix(int y) {
			return foo.fix(y);
		}

		public class Foo {
			private int x = 100;

			public int getX() {
				return x;
			}

			public int fix(int y) {
				// synchronized (this) {
				x = x - y;
				try {
					Thread.sleep(1);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				System.out.println(Thread.currentThread().getName()
						+ " : 当前foo对象的x值= " + x);
				// }
				return x;
			}
		}
	}

	public static class ThreadA {
		public static void testWait() {
			ThreadB b = new ThreadB();
			// 启动计算线程
			b.start();
			// 线程A拥有b对象上的锁。线程为了调用wait()或notify()方法，该线程必须是那个对象锁的拥有者
			synchronized (b) {
				try {
					System.out.println("等待对象b完成计算。。。");
					// 当前线程A等待
					b.wait();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				System.out.println("b对象计算的总和是：" + b.total);
			}
		}

		public static class ThreadB extends Thread {
			int total;

			public void run() {
				synchronized (this) {
					for (int i = 0; i < 101; i++) {
						total += i;
					}
					// （完成计算了）唤醒在此对象监视器上等待的单个线程，在本例中线程A被唤醒
					notify();
				}
			}
		}
	}

	/**
	 * 生产者和消费者
	 */
	public static class Producer extends Thread {

		public static void testProducer() {
			List<String> mBox = new LinkedList<String>();
			Producer producer = new Producer(mBox);
			producer.start();
			for (int i = 0; i < 50; i++) {
				Consumer consumer = new Consumer(mBox, "Consumer#" + i);
				consumer.start();
			}
		}

		private List<String> mBox;

		public Producer(List<String> box) {
			this.mBox = box;
		}

		@Override
		public void run() {
			while (true) {
				synchronized (mBox) {
					mBox.add(System.currentTimeMillis() + "");
					// System.out.println("Producer add a String");
					mBox.notifyAll();
				}
				try {
					Thread.sleep(1);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}

		public static class Consumer extends Thread {
			private List<String> mBox;
			private String consumerName;

			public Consumer(List<String> box, String consumerName) {
				this.mBox = box;
				this.consumerName = consumerName;
			}

			@Override
			public void run() {
				while (true) {
					synchronized (mBox) {
						if (mBox.isEmpty()) {
							try {
								mBox.wait();
							} catch (InterruptedException e) {

							}
						} else {
							String content = mBox.get(0);
							mBox.remove(0);
							System.out.println(consumerName
									+ " consume string : " + content);
						}
					}
					try {
						Thread.sleep(50);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
			}
		}
	}

	/**
	 * 测试线程的join方法
	 */
	public static class TestJoin {
		public static void testJoin() {
			Thread t1 = new MyThread1();
			t1.start();

			for (int i = 0; i < 20; i++) {
				System.out.println("主线程第" + i + "次执行！");
				if (i > 2)
					try {
						// t1线程合并到主线程中，主线程停止执行过程，转而执行t1线程，直到t1执行完毕后继续。
						t1.join();
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
			}
		}

		public static class MyThread1 extends Thread {
			public void run() {
				for (int i = 0; i < 10; i++) {
					System.out.println("线程1第" + i + "次执行！");
				}
			}
		}
	}

	/**
	 * JRE判断程序是否执行结束的标准是所有的前台执线程行完毕了，而不管后台线程的状态. 前台线程是保证执行完毕的，后台线程还没有执行完毕就退出了。
	 * 
	 */
	public static class TestDaemon {
		public static void testDaemon() {
			Thread t1 = new MyCommon();
			Thread t2 = new Thread(new MyDaemon());
			t2.setDaemon(true); // 设置为守护线程

			t2.start();
			t1.start();
		}

		static class MyCommon extends Thread {
			public void run() {
				for (int i = 0; i < 5; i++) {
					System.out.println("线程1第" + i + "次执行！");
					try {
						Thread.sleep(7);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
			}
		}

		static class MyDaemon implements Runnable {
			public void run() {
				for (long i = 0; i < 9999999L; i++) {
					System.out.println("后台线程第" + i + "次执行！");
					try {
						Thread.sleep(7);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
			}
		}
	}

	/**
	 * 生产者和消费者加强版, 出现一个逻辑死锁~~~~~~~ 你妹的
	 * 
	 */
	static class Godown {
		public static void testProducer() {
			final Godown godown = new Godown(30);// 对仓库加锁~~
			Consumer c1 = new Consumer(50, godown);
			Consumer c2 = new Consumer(20, godown);
			Consumer c3 = new Consumer(30, godown);
			Producer p1 = new Producer(10, godown);
			Producer p2 = new Producer(10, godown);
			Producer p3 = new Producer(10, godown);
			Producer p4 = new Producer(10, godown);
			Producer p5 = new Producer(10, godown);
			Producer p6 = new Producer(10, godown);
			Producer p7 = new Producer(80, godown);
			c1.start();
			c2.start();
			c3.start();
			p1.start();
			p2.start();
			p3.start();
			p4.start();
			p5.start();
			p6.start();
			p7.start();
			Thread thread = new Thread(new Runnable() {
				@Override
				public void run() {
					while (true) {
						synchronized (godown) {
							System.out.println("保护线程清理队列啦~~~");
							godown.notifyAll();
						}
						try {
							Thread.sleep(10000);
						} catch (InterruptedException e) {
							e.printStackTrace();
						}
					}
				}
			});
			thread.setDaemon(true);
			thread.start();
		}

		public static final int max_size = 100; // 最大库存量
		public int curnum; // 当前库存量

		Godown() {
		}

		Godown(int curnum) {
			this.curnum = curnum;
		}

		/**
		 * 生产指定数量的产品
		 * 
		 * @param neednum
		 */
		public synchronized void produce(int neednum) {
			// 测试是否需要生产
			while (neednum + curnum > max_size) {
				System.out.println("要生产的产品数量" + neednum + "超过剩余库存量"
						+ (max_size - curnum) + "，暂时不能执行生产任务!");
				try {
					// 当前的生产线程等待
					notifyAll();
					wait();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			// 满足生产条件，则进行生产，这里简单的更改当前库存量
			curnum += neednum;
			System.out.println("已经生产了" + neednum + "个产品，现仓储量为" + curnum);
			// 唤醒在此对象监视器上等待的所有线程
			notifyAll();
		}

		/**
		 * 消费指定数量的产品
		 * 
		 * @param neednum
		 */
		public synchronized void consume(int neednum) {
			// 测试是否可消费
			while (curnum < neednum) {
				try {
					// 当前的消费线程等待
					System.out.println("不够我消费:" + neednum + " 我等!!!");
					notifyAll();
					wait();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			// 满足消费条件，则进行消费，这里简单的更改当前库存量
			curnum -= neednum;
			System.out.println("已经消费了" + neednum + "个产品，现仓储量为" + curnum);
			// 唤醒在此对象监视器上等待的所有线程
			notifyAll();
		}

		static class Producer extends Thread {
			private int neednum; // 生产产品的数量
			private Godown godown; // 仓库

			Producer(int neednum, Godown godown) {
				this.neednum = neednum;
				this.godown = godown;
			}

			public void run() {
				// 生产指定数量的产品
				godown.produce(neednum);
			}
		}

		/**
		 * 消费者
		 */
		static class Consumer extends Thread {
			private int neednum; // 生产产品的数量
			private Godown godown; // 仓库

			Consumer(int neednum, Godown godown) {
				this.neednum = neednum;
				this.godown = godown;
			}

			public void run() {
				// 消费指定数量的产品
				godown.consume(neednum);
			}
		}
	}

	/**
	 * 使用折返锁
	 */
	public static class TestReentrantLock {
		public static void testReentrantLock() {
			// 创建并发访问的账户
			MyCount myCount = new MyCount("95599200901215522", 10000);
			// 创建一个锁对象
			Lock lock = new ReentrantLock();
			// 创建一个线程池
			ExecutorService pool = Executors.newCachedThreadPool();
			// 创建一些并发访问用户，一个信用卡，存的存，取的取，好热闹啊
			User u1 = new User("张三", myCount, -4000, lock);
			User u2 = new User("张三他爹", myCount, 6000, lock);
			User u3 = new User("张三他弟", myCount, -8000, lock);
			User u4 = new User("张三", myCount, 800, lock);
			// 在线程池中执行各个用户的操作
			pool.execute(u1);
			pool.execute(u2);
			pool.execute(u3);
			pool.execute(u4);
			// 关闭线程池
			pool.shutdown();
		}

		/**
		 * 信用卡的用户
		 */
		static class User implements Runnable {
			private String name; // 用户名
			private MyCount myCount; // 所要操作的账户
			private int iocash; // 操作的金额，当然有正负之分了
			private Lock myLock; // 执行操作所需的锁对象

			User(String name, MyCount myCount, int iocash, Lock myLock) {
				this.name = name;
				this.myCount = myCount;
				this.iocash = iocash;
				this.myLock = myLock;
			}

			public void run() {
				// 获取锁
				myLock.lock();
				// 执行现金业务
				System.out.println(name + "正在操作" + myCount + "账户，金额为" + iocash
						+ "，当前金额为" + myCount.getCash());
				myCount.setCash(myCount.getCash() + iocash);
				System.out.println(name + "操作" + myCount + "账户成功，金额为" + iocash
						+ "，当前金额为" + myCount.getCash());
				// 释放锁，否则别的线程没有机会执行了
				myLock.unlock();
			}
		}

		/**
		 * 信用卡账户，可随意透支
		 */
		static class MyCount {
			private String oid; // 账号
			private int cash; // 账户余额

			MyCount(String oid, int cash) {
				this.oid = oid;
				this.cash = cash;
			}

			public String getOid() {
				return oid;
			}

			public void setOid(String oid) {
				this.oid = oid;
			}

			public int getCash() {
				return cash;
			}

			public void setCash(int cash) {
				this.cash = cash;
			}

			@Override
			public String toString() {
				return "MyCount{" + "oid='" + oid + '\'' + ", cash=" + cash
						+ '}';
			}
		}
	}

	/**
	 *	使用读写锁
	 */
	public static class TestReentrantReadWriteLock {
		public static void testReentrantReadWriteLock() {
			// 创建并发访问的账户
			MyCount myCount = new MyCount("95599200901215522", 10000);
			// 创建一个锁对象
			ReadWriteLock lock = new ReentrantReadWriteLock(false);
			// 创建一个线程池
			ExecutorService pool = Executors.newFixedThreadPool(2);
			// 创建一些并发访问用户，一个信用卡，存的存，取的取，好热闹啊
			User u1 = new User("张三", myCount, -4000, lock, false);
			User u2 = new User("张三他爹", myCount, 6000, lock, false);
			User u3 = new User("张三他弟", myCount, -8000, lock, false);
			User u4 = new User("张三", myCount, 800, lock, false);
			User u5 = new User("张三他爹", myCount, 0, lock, true);
			// 在线程池中执行各个用户的操作
			pool.execute(u1);
			pool.execute(u2);
			pool.execute(u3);
			pool.execute(u4);
			pool.execute(u5);
			// 关闭线程池
			pool.shutdown();
		}
		
		/** 
		* 信用卡的用户 
		*/ 
		static class User implements Runnable { 
		        private String name;                //用户名 
		        private MyCount myCount;        //所要操作的账户 
		        private int iocash;                 //操作的金额，当然有正负之分了 
		        private ReadWriteLock myLock;                //执行操作所需的锁对象 
		        private boolean ischeck;        //是否查询 

		        User(String name, MyCount myCount, int iocash, ReadWriteLock myLock, boolean ischeck) { 
		                this.name = name; 
		                this.myCount = myCount; 
		                this.iocash = iocash; 
		                this.myLock = myLock; 
		                this.ischeck = ischeck; 
		        } 

		        public void run() { 
		                if (ischeck) { 
		                        //获取读锁 
		                        myLock.readLock().lock(); 
		                        System.out.println("读：" + name + "正在查询" + myCount + "账户，当前金额为" + myCount.getCash()); 
		                        //释放读锁 
		                        myLock.readLock().unlock(); 
		                } else { 
		                        //获取写锁 
		                        myLock.writeLock().lock(); 
		                        //执行现金业务 
		                        System.out.println("写：" + name + "正在操作" + myCount + "账户，金额为" + iocash + "，当前金额为" + myCount.getCash()); 
		                        myCount.setCash(myCount.getCash() + iocash); 
		                        System.out.println("写：" + name + "操作" + myCount + "账户成功，金额为" + iocash + "，当前金额为" + myCount.getCash()); 
		                        //释放写锁 
		                        myLock.writeLock().unlock(); 
		                } 
		        } 
		} 

		/** 
		* 信用卡账户，可随意透支 
		*/ 
		static class MyCount { 
		        private String oid;         //账号 
		        private int cash;             //账户余额 

		        MyCount(String oid, int cash) { 
		                this.oid = oid; 
		                this.cash = cash; 
		        } 

		        public String getOid() { 
		                return oid; 
		        } 

		        public void setOid(String oid) { 
		                this.oid = oid; 
		        } 

		        public int getCash() { 
		                return cash; 
		        } 

		        public void setCash(int cash) { 
		                this.cash = cash; 
		        } 

		        @Override 
		        public String toString() { 
		                return "MyCount{" + 
		                                "oid='" + oid + '\'' + 
		                                ", cash=" + cash + 
		                                '}'; 
		        } 
		}
	}
}
