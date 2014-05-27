package org.jabe.neverland.compute;

public class Recursive {

	private static final int N = 93;

	public static long computeRecursively(int n) {
		if (n > 1)
			return computeRecursively(n - 2) + computeRecursively(n - 1);
		return n;
	}

	public static long computeRecursivelyWithLoop(int n) {
		if (n > 1) {
			long result = 1;
			do {
				result += computeRecursivelyWithLoop(n - 2);
				n--;
			} while (n > 1);
			return result;
		}
		return n;
	}

	public static long computeIteratively(int n) {
		if (n > 1) {
			long a = 0, b = 1;
			do {
				long tmp = b;
				b += a;
				a = tmp;
			} while (--n > 1);
			return b;
		}
		return n;
	}

	public static long computeIterativelyFaster(int n) {
		if (n > 1) {
			long a, b = 1;
			n--;
			a = n & 1;
			n /= 2;
			while (n-- > 0) {
				a += b;
				b += a;
			}
			return b;
		}
		return n;
	}

	public static void main(String[] args) {
//		computeTime(new Compute() {
//
//			@Override
//			public String getName() {
//				return "computeRecursively";
//			}
//
//			@Override
//			public void compute() {
//				computeRecursively(N);
//			}
//		});
//		computeTime(new Compute() {
//
//			@Override
//			public String getName() {
//				return "computeRecursivelyWithLoop";
//			}
//
//			@Override
//			public void compute() {
//				computeRecursivelyWithLoop(N);
//			}
//		});

		computeTime(new Compute() {

			@Override
			public String getName() {
				return "computeIteratively";
			}

			@Override
			public long compute() {
				return computeIteratively(N);
			}
		});

		computeTime(new Compute() {

			@Override
			public String getName() {
				return "computeIterativelyFaster";
			}

			@Override
			public long compute() {
				return computeIterativelyFaster(N);
			}
		});
	}

	public interface Compute {
		public long compute();

		public String getName();
	}

	public static void computeTime(Compute c) {
		System.out.println("start : " + c.getName());
		long start = System.currentTimeMillis();
		System.out.println("end : " + c.getName() + c.compute() + " cast : "
				+ (System.currentTimeMillis() - start));
	}
}
