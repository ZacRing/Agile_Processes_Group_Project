from django import forms

class HeartDiseaseForm(forms.Form):
    age_years = forms.IntegerField(
        label="Age (years)",
        min_value=1,
        max_value=120
    )

    is_male = forms.ChoiceField(
        label="Sex",
        choices=[
            ("1", "Male"),
            ("0", "Female"),
        ],
        widget=forms.RadioSelect
    )

    height = forms.FloatField(
        label="Height (cm)",
        min_value=50,
        max_value=250
    )

    weight = forms.FloatField(
        label="Weight (kg)",
        min_value=10,
        max_value=300
    )

    ap_hi = forms.IntegerField(
        label="Systolic blood pressure (ap_hi)",
        min_value=50,
        max_value=300
    )

    ap_lo = forms.IntegerField(
        label="Diastolic blood pressure (ap_lo)",
        min_value=30,
        max_value=200
    )

    cholesterol = forms.ChoiceField(
        label="Cholesterol level",
        choices=[
            ("1", "Normal"),
            ("2", "Above normal"),
            ("3", "Well above normal"),
        ]
    )

    gluc = forms.ChoiceField(
        label="Glucose level",
        choices=[
            ("1", "Normal"),
            ("2", "Above normal"),
            ("3", "Well above normal"),
        ]
    )

    smoke = forms.ChoiceField(
        label="Do you smoke?",
        choices=[
            ("1", "Yes"),
            ("0", "No"),
        ],
        widget=forms.RadioSelect
    )

    alco = forms.ChoiceField(
        label="Do you consume alcohol?",
        choices=[
            ("1", "Yes"),
            ("0", "No"),
        ],
        widget=forms.RadioSelect
    )

    active = forms.ChoiceField(
        label="Are you physically active?",
        choices=[
            ("1", "Yes"),
            ("0", "No"),
        ],
        widget=forms.RadioSelect
    )