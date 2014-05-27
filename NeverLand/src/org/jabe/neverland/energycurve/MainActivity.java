package org.jabe.neverland.energycurve;

import java.util.ArrayList;

import org.jabe.neverland.R;

import android.app.Activity;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.view.Menu;

public class MainActivity extends Activity {

    private EnergyCurveView erenergyCurve;
    private DisplayMetrics dm = new DisplayMetrics();

	@Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        erenergyCurve = (EnergyCurveView) findViewById(R.id.erenergycurve);
        getWindowManager().getDefaultDisplay().getMetrics(dm);
        erenergyCurve.setWindowsWH(dm);
        ArrayList<EnergyItem> energys = new ArrayList<EnergyItem>();
        energys.add(new EnergyItem("1", 0.0f, "无"));
        energys.add(new EnergyItem("2", 0.0f, "无"));
        energys.add(new EnergyItem("3", 8.0f, "无"));
        energys.add(new EnergyItem("4", 6.0f, "无"));
        energys.add(new EnergyItem("5", 2.8f, "无"));
        energys.add(new EnergyItem("6", 0.0f, "无"));
        energys.add(new EnergyItem("7", 10.0f, "无"));
        erenergyCurve.setData(energys, EnergyType.DAY);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        return true;
    }
}
